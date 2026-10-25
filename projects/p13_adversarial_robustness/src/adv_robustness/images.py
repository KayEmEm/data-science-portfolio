"""Evasion attacks on the road damage detector of project 6, and input transformation defenses.

Every attack raises the training loss of the detector on the true boxes (box, class and
distribution focal loss of Ultralytics), so that damage is missed, misplaced or misclassified and
false detections appear. The attacker has the model weights (white box):

* FGSM and PGD: a perturbation of every pixel, at most ``epsilon`` in the infinity norm;
* a universal patch: one square of ``patch_size`` pixels, trained on the training images and pasted
  at a random place on each test image, as a sticker on the road would be. A patch of random noise
  of the same size, at the same places, is the control for plain occlusion.

Defenses transform every input before the detector: JPEG compression, a median filter, and
Gaussian noise. The attacks do not know the defense (no adaptive attack), so the defended numbers
are an upper bound on robustness. Adversarial training of the detector is not run: one training
run of project 6 takes about 10 hours on the CPU of the build environment.

Scores are mAP50 of the Ultralytics validator on the test images of the sample, as in project 6.
"""

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from portfolio_common.seed import set_seed
from road_damage.config import CLASS_IDS

from adv_robustness.config import (
    ARTIFACTS_DIR,
    CONF,
    DATA_DIR,
    DETECTOR_WEIGHTS,
    IMAGE_SIZE,
    SEED,
    ImageBudget,
)

DEFENSES = ("none", "jpeg", "median", "noise")
JPEG_QUALITY = 75
MEDIAN_SIZE = 3
NOISE_SIGMA = 8 / 255
BATCH = 8


def load_detector(weights: Path):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    yolo = YOLO(str(weights))
    net = yolo.model.float().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    # The loss reads its gains (box, cls, dfl) from the training arguments stored with the weights.
    stored = net.args if isinstance(net.args, dict) else vars(net.args)
    net.args = get_cfg(overrides={k: v for k, v in stored.items() if k in ("box", "cls", "dfl")})
    return yolo, net


def split_paths(yolo_dir: Path, split: str) -> list[Path]:
    return [yolo_dir / line for line in (yolo_dir / f"{split}.txt").read_text().split()]


def label_path(image: Path, yolo_dir: Path, split: str) -> Path:
    return yolo_dir / "labels" / split / f"{image.stem}.txt"


def targets(labels: list[Path]) -> dict:
    """True boxes of a batch in the format of the Ultralytics loss (normalized centre and size)."""
    cls, boxes, index = [], [], []
    for i, path in enumerate(labels):
        rows = np.loadtxt(path, ndmin=2) if path.exists() and path.stat().st_size else np.zeros((0, 5))
        cls += rows[:, 0].tolist()
        boxes += rows[:, 1:].tolist()
        index += [i] * len(rows)
    return {
        "cls": torch.tensor(cls, dtype=torch.float32).view(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).view(-1, 4),
        "batch_idx": torch.tensor(index, dtype=torch.float32),
    }


def attack_loss(net, x: torch.Tensor, truth: dict) -> torch.Tensor:
    """The negative detection loss: lowering it raises the loss on the true boxes."""
    loss, _ = net.loss({**truth, "img": x})
    return -loss.sum()


def read_images(paths: list[Path], size: int) -> torch.Tensor:
    arrays = [
        np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
        for p in paths
    ]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def fgsm(net, x: torch.Tensor, truth: dict, epsilon: float) -> torch.Tensor:
    x = x.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(attack_loss(net, x, truth), x)
    return (x - epsilon * grad.sign()).clamp(0, 1).detach()


def pgd(net, x: torch.Tensor, truth: dict, epsilon: float, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    step = epsilon / 4
    noise = (torch.rand(x.shape, generator=generator) * 2 - 1) * epsilon
    adv = (x + noise).clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        (grad,) = torch.autograd.grad(attack_loss(net, adv, truth), adv)
        adv = adv.detach() - step * grad.sign()
        adv = torch.min(torch.max(adv, x - epsilon), x + epsilon).clamp(0, 1)
    return adv.detach()


def places(n: int, side: int, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, side - size + 1, size=(n, 2))


def paste(x: torch.Tensor, patch: torch.Tensor, where: np.ndarray) -> torch.Tensor:
    """Each image with the patch at its (top, left) place."""
    out = x.clone()
    size = patch.shape[-1]
    for i, (top, left) in enumerate(where):
        out[i, :, top : top + size, left : left + size] = patch
    return out


def train_patch(
    net, paths: list[Path], labels: list[Path], budget: ImageBudget, size: int, seed: int, steps: int
) -> torch.Tensor:
    """A universal patch that raises the detection loss on the training images."""
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    patch = torch.rand(3, budget.patch_size, budget.patch_size, generator=generator).requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=budget.patch_lr)
    for _ in range(steps):
        chosen = rng.choice(len(paths), size=min(budget.patch_batch, len(paths)), replace=False)
        x = read_images([paths[i] for i in chosen], size)
        truth = targets([labels[i] for i in chosen])
        optimizer.zero_grad()
        where = places(len(x), size, budget.patch_size, rng)
        loss = attack_loss(net, paste(x, patch.clamp(0, 1), where), truth)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    return patch.detach()


def defend(image: Image.Image, defense: str, rng: np.random.Generator) -> Image.Image:
    if defense == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if defense == "median":
        return image.filter(ImageFilter.MedianFilter(MEDIAN_SIZE))
    if defense == "noise":
        array = np.asarray(image, dtype=np.float32) / 255
        array = np.clip(array + rng.normal(0, NOISE_SIGMA, array.shape), 0, 1)
        return Image.fromarray(np.round(array * 255).astype(np.uint8))
    return image


def to_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.round(tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def write_set(
    root: Path, images: list[tuple[str, torch.Tensor]], labels: dict[str, Path], defense: str, seed: int
) -> Path:
    """A test set in the Ultralytics layout: PNG images (lossless), their labels and a dataset file."""
    rng = np.random.default_rng(seed)
    shutil.rmtree(root, ignore_errors=True)
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    for stem, tensor in images:
        defend(to_image(tensor), defense, rng).save(root / "images" / "test" / f"{stem}.png")
        shutil.copy(labels[stem], root / "labels" / "test" / f"{stem}.txt")
    spec = {
        "path": str(root.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {i: code for code, i in CLASS_IDS.items()},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return root / "dataset.yaml"


def scores(yolo, dataset: Path, size: int) -> dict:
    metrics = yolo.val(
        data=str(dataset),
        split="test",
        imgsz=size,
        batch=16,
        conf=CONF,
        device="cpu",
        plots=False,
        verbose=False,
        workers=0,
        project=str(dataset.parent / "val"),
        name="run",
        exist_ok=True,
    )
    return {
        "map50": float(metrics.box.map50),
        "recall": float(metrics.box.mr),
        "precision": float(metrics.box.mp),
    }


def attacked_sets(net, paths, labels, patch, noise_patch, budget: ImageBudget, size: int, steps: int) -> dict:
    """Every test image under every attack, as (stem, tensor) pairs per attack."""
    rng = np.random.default_rng(SEED)
    sets: dict[str, list] = {}
    for start in range(0, len(paths), BATCH):
        chunk = paths[start : start + BATCH]
        stems = [p.stem for p in chunk]
        x = read_images(chunk, size)
        truth = targets([labels[stem] for stem in stems])
        where = places(len(x), size, patch.shape[-1], rng)
        batch = {
            "clean": x,
            "noise_patch": paste(x, noise_patch, where),
            "adversarial_patch": paste(x, patch, where),
        }
        for epsilon in budget.epsilons:
            batch[f"fgsm_{epsilon * 255:.0f}"] = fgsm(net, x, truth, epsilon)
            batch[f"pgd_{epsilon * 255:.0f}"] = pgd(net, x, truth, epsilon, steps, SEED + start)
        for name, tensor in batch.items():
            sets.setdefault(name, []).extend(zip(stems, tensor, strict=True))
    return sets


def run(
    data_dir: Path,
    artifacts: Path,
    weights: Path = DETECTOR_WEIGHTS,
    size: int = IMAGE_SIZE,
    quick: bool = False,
    budget: ImageBudget | None = None,
) -> dict:
    set_seed(SEED)
    torch.manual_seed(SEED)
    budget = budget or ImageBudget()
    yolo_dir = data_dir / "images" / "yolo"
    test_paths, train_paths = split_paths(yolo_dir, "test"), split_paths(yolo_dir, "train")
    labels = {p.stem: label_path(p, yolo_dir, "test") for p in test_paths}
    train_labels = [label_path(p, yolo_dir, "train") for p in train_paths]
    yolo, net = load_detector(weights)
    budget = ImageBudget(
        **{**budget.__dict__, "patch_size": max(8, round(budget.patch_size * size / IMAGE_SIZE))}
    )

    patch = train_patch(
        net, train_paths, train_labels, budget, size, SEED, 3 if quick else budget.patch_steps
    )
    noise_patch = torch.rand(patch.shape, generator=torch.Generator().manual_seed(SEED + 1))
    sets = attacked_sets(
        net, test_paths, labels, patch, noise_patch, budget, size, 2 if quick else budget.pgd_steps
    )

    out = artifacts / "images"
    out.mkdir(parents=True, exist_ok=True)
    to_image(patch).resize((256, 256), Image.NEAREST).save(out / "patch.png")
    # One image with boxes, under each attack, for the report figure.
    with_boxes = next((i for i, (stem, _) in enumerate(sets["clean"]) if labels[stem].stat().st_size), 0)
    for name, images in sets.items():
        to_image(images[with_boxes][1]).save(out / f"example_{name}.png")
    (out / "example.json").write_text(json.dumps({"stem": sets["clean"][with_boxes][0]}))

    summary = {
        "test_images": len(test_paths),
        "test_images_with_damage": int(sum(p.stat().st_size > 0 for p in labels.values())),
        "train_images": len(train_paths),
        "budget": budget.__dict__,
        "weights": weights.name,
        "image_size": size,
        "results": {},
    }
    for name, images in sets.items():
        summary["results"][name] = {}
        for defense in DEFENSES:
            dataset = write_set(out / "sets" / f"{name}_{defense}", images, labels, defense, SEED)
            summary["results"][name][defense] = scores(yolo, dataset, size)
        line = ", ".join(f"{d} {summary['results'][name][d]['map50']:.3f}" for d in DEFENSES)
        print(f"images {name}: mAP50 {line}")
    shutil.rmtree(out / "sets", ignore_errors=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--weights", type=Path, default=DETECTOR_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--quick", action="store_true", help="few attack steps, for tests and the smoke run")
    args = parser.parse_args()
    run(args.data_dir, args.artifacts, args.weights, args.imgsz, args.quick)


if __name__ == "__main__":
    main()
