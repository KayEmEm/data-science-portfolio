# Segment playbook

One page per segment of the selected k means solution (RFM and product mix features, quantile scaling, k = 3), for the 5,240 customers with a positive net spend before the cutoff of 9 September 2011. Values before the cutoff describe the segment; values after it (10 September to 9 December 2011) show what the segment did next. The suggested actions follow from these numbers and have not been tested in a campaign; each one names the measure that would show whether it works.

Inside each segment, the ten groups of the RFM grid rank customers better than the segment alone (the grid explains 26 percent of the variance of next quarter's log spend, the segments 21 percent), so the tables below list them as the second level of targeting.

## Segment 1: Active repeat buyers

| Measure | Value |
| --- | --- |
| Customers | 1,508 (29 percent) |
| Net spend before the cutoff | 10.8 million GBP (82 percent) |
| Median orders, days since last order, tenure | 9 orders, 43 days, 591 days |
| Median basket, distinct products | 384 GBP, 132 products |
| Product mix | the broadest range; bags 10 percent of value (other segments 7 percent) |
| Bought next quarter | 73 percent |
| Spend next quarter | 2.13 million GBP (77 percent of all), mean 1,413 GBP |
| RFM grid groups inside | champions 654, loyal 640, at risk 79, potential loyalists 55, cannot lose 47 |

**What it is:** regular customers with large baskets (median 384 GBP), a pattern that fits trade buyers who reorder every few weeks. They carry the business: almost four in five pounds of next quarter's spend came from them.

**Suggested actions:**

* Keep them: account contact, early access to the Christmas range in late summer, and order reminders timed to each customer's usual gap between orders.
* Watch the 126 customers that the grid marks as at risk or cannot lose: they ordered often but not recently. A personal contact costs little against their past spend.
* Avoid broad discounts: 73 percent bought again without one.

**Measure:** share of the segment that orders within 60 days, and spend per customer, against a held out group that receives the standard emails only.

## Segment 2: Occasional repeat buyers

| Measure | Value |
| --- | --- |
| Customers | 2,169 (41 percent) |
| Net spend before the cutoff | 1.87 million GBP (14 percent) |
| Median orders, days since last order, tenure | 3 orders, 182 days, 455 days |
| Median basket, distinct products | 227 GBP, 38 products |
| Product mix | close to the average; lights and candles 11 percent of value |
| Bought next quarter | 40 percent |
| Spend next quarter | 0.47 million GBP (17 percent of all), mean 218 GBP |
| RFM grid groups inside | at risk 635, potential loyalists 435, loyal 377, hibernating 263, need attention 228 |

**What it is:** customers who came back a few times but at long intervals; half of them last ordered more than six months before the cutoff. It is the largest segment and the one where a change in behavior would move the most revenue.

**Suggested actions:**

* Seasonal reactivation before the Christmas buying period, with the categories each customer bought before.
* A threshold offer (for example free delivery above a basket value) to raise order frequency rather than price discounts.
* Treat the 435 potential loyalists and 377 loyal customers as candidates to move into segment 1; treat the 263 hibernating customers like segment 3.

**Measure:** share that orders within 90 days and orders per customer over the next half year, against a held out group.

## Segment 3: Single order customers

| Measure | Value |
| --- | --- |
| Customers | 1,563 (30 percent) |
| Net spend before the cutoff | 0.53 million GBP (4 percent) |
| Median orders, days since last order, tenure | 1 order, 322 days, 322 days |
| Median basket, distinct products | 234 GBP, 16 products |
| Product mix | close to the average; Christmas 4 percent of value (segment 1: 3 percent) |
| Bought next quarter | 21 percent |
| Spend next quarter | 0.16 million GBP (6 percent of all), mean 105 GBP |
| RFM grid groups inside | hibernating 1,051, about to sleep 275, promising 95, new customers 76, potential loyalists 66 |

**What it is:** every customer with exactly one order before the cutoff; for half of them that order was more than ten months earlier. One in five still bought in the next quarter, which is the Christmas season.

**Suggested actions:**

* Low cost channels only (email), with one reminder before the Christmas season; stop paid contact for customers without an order for more than a year.
* Give the 171 new and promising customers (one order, recency score 4 or 5 in the grid) a welcome sequence: their second order decides whether they move to segment 2.
* Ask a sample why they did not return; the data hold no reason.

**Measure:** share with a second order within 90 days of the contact, and cost per second order.

## Notes

* The segments are mostly defined by recency, frequency and spend; the product mix differs by at most 3.4 percentage points between segments and should not drive segment specific assortments.
* Return share was left out of the clustering after the first attempt (see the README); the mean return share is 1 to 2 percent of purchase value in every segment.
* The segments describe customers up to 9 September 2011. Before use, the pipeline should be run again on current data, and a customer's segment refreshed every quarter.
