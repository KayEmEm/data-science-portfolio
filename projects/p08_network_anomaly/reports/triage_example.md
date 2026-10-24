# Alert triage example: Isolation Forest

At the threshold set on validation flows (1% of normal flows flagged), the detector raises 10,462 alerts on 82,332 test flows; 9,959 of them are attacks (precision 0.952).

An analyst does not read 10,462 alerts one by one. They fall into 31 groups of the same protocol, service and state with the same three most unusual features (the features whose signed log value lies furthest from the training sample median, in interquartile ranges). The 15 groups with the highest scores are below. The share of attacks and the labels are shown for this example only; the detector does not see them.

| Group | Alerts | Top score | Protocol / service / state | Most unusual features: group median (training median) | Share of attacks | Labels |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 1 | 6,554 | 0.6538 | udp / dns / int | ct_dst_ltm 26 (3); ct_dst_sport_ltm 21 (1); ct_src_dport_ltm 25 (1) | 1.00 | Generic 6,554 |
| 2 | 45 | 0.6451 | arp / - / con | dinpkt 1.552e+04 (0.902); dur 30.99 (0.03814); sttl 0 (31) | 0.00 | Normal 45 |
| 3 | 2 | 0.6413 | tcp / http / con | ackdat 0.06759 (0.000125); synack 0.03305 (0.000515); tcprtt 0.1006 (0.000644) | 1.00 | DoS 1, Exploits 1 |
| 4 | 27 | 0.6334 | tcp / - / con | ackdat 0.5667 (0.000125); synack 1.264 (0.000515); tcprtt 2.282 (0.000644) | 0.07 | Normal 25, DoS 2 |
| 5 | 3,139 | 0.6307 | udp / dns / int | ct_dst_sport_ltm 17 (1); ct_src_dport_ltm 17 (1); sttl 254 (31) | 1.00 | Generic 3,139 |
| 6 | 81 | 0.6275 | udp / dns / int | ct_dst_sport_ltm 9 (1); ct_dst_src_ltm 51 (2); ct_src_dport_ltm 11 (1) | 1.00 | Generic 81 |
| 7 | 32 | 0.6215 | tcp / - / req | ct_dst_ltm 32 (3); ct_src_ltm 33 (3); dur 8.432 (0.03814) | 0.00 | Normal 32 |
| 8 | 197 | 0.6208 | tcp / - / req | ct_dst_ltm 22 (3); dur 9.352 (0.03814); sttl 254 (31) | 0.00 | Normal 197 |
| 9 | 6 | 0.6208 | udp / dns / req | ct_state_ttl 6 (0); dur 46.09 (0.03814); sttl 254 (31) | 1.00 | Fuzzers 6 |
| 10 | 9 | 0.6206 | tcp / smtp / con | ackdat 0.07015 (0.000125); synack 0.06337 (0.000515); tcprtt 0.1442 (0.000644) | 1.00 | Exploits 6, DoS 3 |
| 11 | 3 | 0.6169 | udp / - / con | ct_dst_sport_ltm 3 (1); dur 59.82 (0.03814); sttl 254 (31) | 1.00 | DoS 3 |
| 12 | 22 | 0.6156 | tcp / http / fin | ackdat 0.06801 (0.000125); synack 0.08264 (0.000515); tcprtt 0.15 (0.000644) | 0.32 | Normal 15, Exploits 5, Fuzzers 2 |
| 13 | 2 | 0.6155 | arp / - / con | dinpkt 2.573e+04 (0.902); sload 8.704 (4.327e+05); sttl 0 (31) | 0.00 | Normal 2 |
| 14 | 4 | 0.6124 | udp / - / con | dur 57.07 (0.03814); smean 653 (73); sttl 254 (31) | 1.00 | DoS 4 |
| 15 | 109 | 0.6123 | ospf / - / req | ct_state_ttl 6 (0); dur 58.42 (0.03814); sttl 254 (31) | 1.00 | Exploits 50, DoS 32, Fuzzers 12, Reconnaissance 10, Backdoor 4, Generic 1 |

Attack types among all alerts: Generic 9,799, Exploits 63, DoS 46, Fuzzers 37, Reconnaissance 10, Backdoor 4.
