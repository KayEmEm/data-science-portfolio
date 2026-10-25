import numpy as np
import torch
from PIL import Image

from adv_robustness import images, report


def test_targets_format(tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("1 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.05 0.05\n")
    second.write_text("")
    truth = images.targets([first, second, tmp_path / "missing.txt"])
    assert truth["cls"].shape == (2, 1) and truth["bboxes"].shape == (2, 4)
    assert truth["batch_idx"].tolist() == [0.0, 0.0]


def test_paste_and_defenses():
    x = torch.zeros(2, 3, 32, 32)
    patch = torch.ones(3, 8, 8)
    out = images.paste(x, patch, np.array([[0, 0], [24, 24]]))
    assert out[0, :, :8, :8].min() == 1 and out[1, :, 24:, 24:].min() == 1
    assert out.sum() == 2 * 3 * 64
    picture = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    rng = np.random.default_rng(0)
    for defense in images.DEFENSES:
        assert images.defend(picture, defense, rng).size == (32, 32)


def test_attacks_stay_in_budget_and_pipeline_runs(sources, data_dir, tmp_path):
    yolo, net = images.load_detector(sources / "detector" / "untrained.pt")
    yolo_dir = data_dir / "images" / "yolo"
    paths = images.split_paths(yolo_dir, "test")[:2]
    x = images.read_images(paths, 64)
    truth = images.targets([images.label_path(p, yolo_dir, "test") for p in paths])
    epsilon = 4 / 255
    for adv in (images.fgsm(net, x, truth, epsilon), images.pgd(net, x, truth, epsilon, 2, 0)):
        assert (adv - x).abs().max() <= epsilon + 1e-6
        assert adv.min() >= 0 and adv.max() <= 1

    summary = images.run(data_dir, tmp_path, sources / "detector" / "untrained.pt", 64, quick=True)
    assert set(summary["results"]["clean"]) == set(images.DEFENSES)
    assert (tmp_path / "images" / "patch.png").exists()
    grid, defense = report.image_tables(summary)
    assert len(grid) == len(summary["results"]) and len(defense) == 2 * (len(images.DEFENSES) - 1)
