xgb_clf = models.XGBoostMutliClass(nthread=8, eta=0.1, max_depth=20, num_round=100, silent=1)

Private Score: 0.69731

3360 features
including finelline number with occurences above 0.3 quantile

xgb_clf = models.XGBoostMutliClass(nthread=4, eta=0.1, max_depth=20, num_round=50, silent=1)
1. feature 3 : FinelineNumber_has_max (0.143909)
2. feature 0 : ScanCountSum (0.086384)
3. feature 5 : Purchased_num (0.073598)
4. feature 1 : Encoded_UPC_nunique (0.060439)
5. feature 23 : DSD GROCERY (0.036789)
6. feature 26 : FINANCIAL SERVICES (0.031983)
7. feature 63 : PRODUCE (0.027422)
8. feature 56 : PERSONAL CARE (0.027124)
9. feature 58 : PHARMACY OTC (0.026834)
10. feature 30 : GROCERY DRY GOODS (0.026183)
11. feature 4 : Returned_num (0.025144)
12. feature 22 : DAIRY (0.022426)
13. feature 66 : SERVICE DELI (0.021750)
14. feature 38 : IMPULSE MERCHANDISE (0.019673)
15. feature 49 : MENS WEAR (0.018560)
16. feature 2 : FinelineNumber_max_num (0.018249)
17. feature 36 : HOUSEHOLD CHEMICALS/SUPP (0.016971)
18. feature 40 : INFANT CONSUMABLE HARDLINES (0.014611)
19. feature 11 : BEAUTY (0.013655)
20. feature 27 : FROZEN FOODS (0.012693)
21. feature 37 : HOUSEHOLD PAPER GOODS (0.012370)
22. feature 59 : PHARMACY RX (0.012093)
23. feature 17 : CANDY, TOBACCO, COOKIES (0.011499)
24. feature 19 : COMM BREAD (0.011098)
25. feature 47 : MEAT - FRESH & FROZEN (0.010723)
26. feature 43 : LADIESWEAR (0.010170)
27. feature 57 : PETS AND SUPPLIES (0.010146)
28. feature 8 : AUTOMOTIVE (0.009603)
29. feature 46 : LIQUOR,WINE,BEER (0.009442)
30. feature 18 : CELEBRATION (0.009232)
31. feature 68 : SHOES (0.009146)
32. feature 34 : HOME MANAGEMENT (0.008807)
33. feature 9 : BAKERY (0.008718)
34. feature 72 : TOYS (0.008495)
35. feature 21 : COOK AND DINE (0.008200)
36. feature 70 : SPORTING GOODS (0.008101)
37. feature 51 : OFFICE SUPPLIES (0.007944)
38. feature 45 : LAWN AND GARDEN (0.007746)
39. feature 31 : HARDWARE (0.007276)
40. feature 24 : ELECTRONICS (0.007265)
41. feature 62 : PRE PACKED DELI (0.006162)
42. feature 73 : WIRELESS (0.005712)
43. feature 33 : HOME DECOR (0.005338)
44. feature 25 : FABRICS AND CRAFTS (0.005311)
45. feature 39 : INFANT APPAREL (0.005013)
46. feature 10 : BATH AND SHOWER (0.004736)
47. feature 12 : BEDDING (0.004387)
48. feature 14 : BOYS WEAR (0.004269)
49. feature 41 : JEWELRY AND SUNGLASSES (0.004191)
50. feature 29 : GIRLS WEAR, 4-6X  AND 7-14 (0.004086)
51. feature 35 : HORTICULTURE AND ACCESS (0.004071)
52. feature 48 : MEDIA AND GAMING (0.003809)
53. feature 52 : OPTICAL - FRAMES (0.003351)
54. feature 69 : SLEEPWEAR/FOUNDATIONS (0.003279)
55. feature 55 : PAINT AND ACCESSORIES (0.002760)
56. feature 15 : BRAS & SHAPEWEAR (0.002557)
57. feature 64 : SEAFOOD (0.002335)
58. feature 7 : ACCESSORIES (0.002274)
59. feature 13 : BOOKS AND MAGAZINES (0.001758)
60. feature 60 : PLAYERS AND ELECTRONICS (0.001712)
61. feature 28 : FURNITURE (0.001557)
62. feature 6 : 1-HR PHOTO (0.001466)
63. feature 71 : SWIMWEAR/OUTERWEAR (0.001316)
64. feature 42 : LADIES SOCKS (0.001179)
65. feature 61 : PLUS AND MATERNITY (0.000827)
66. feature 44 : LARGE HOUSEHOLD GOODS (0.000723)
67. feature 67 : SHEER HOSIERY (0.000707)
68. feature 16 : CAMERAS AND SUPPLIES (0.000664)
69. feature 50 : MENSWEAR (0.000654)
70. feature 53 : OPTICAL - LENSES (0.000558)
71. feature 20 : CONCEPT STORES (0.000420)
72. feature 54 : OTHER DEPARTMENTS (0.000234)
73. feature 65 : SEASONAL (0.000108)
74. feature 32 : HEALTH AND BEAUTY AIDS (0.000005)

Log loss 0.82194



XGBoostMutliClass(eta=0.1, max_depth=6, nthread=4, num_round=5, silent=1)
1. feature 11 : Encoded_DepartmentDescription_has_max_by_vn (0.135551)
2. feature 13 : ScanCount_purchased_items_by_vn (0.129917)
3. feature 9 : FinelineNumber_has_max_by_vn (0.124469)
4. feature 7 : Encoded_Upc_nunique_groupby_vn (0.096143)
5. feature 6 : FinelineNumber_nunique_groupby_vn (0.080364)
6. feature 10 : Encoded_DepartmentDescription_max_by_vn (0.066343)
7. feature 5 : Encoded_DepartmentDescription_nunique_groupby_vn (0.065772)
8. feature 4 : Encoded_Weekday (0.064907)
9. feature 3 : Encoded_Upc (0.060493)
10. feature 2 : Encoded_DepartmentDescription (0.057254)
11. feature 1 : FinelineNumber (0.053116)
12. feature 8 : FinelineNumber_max_by_vn (0.032754)
13. feature 12 : ScanCount_returned_items_by_vn (0.021712)
14. feature 0 : ScanCount (0.011204)

classification accuracy : 0.631135
Log loss: 2.17541434135

RandomForestClassifier(n_estimators=100, n_jobs=4)
Feature ranking:
1.  feature 5  :  ScanCount_sum_groupby_vn (0.146234)
2.  feature 12 :  Encoded_DepartmentDescription_has_max_by_vn (0.136101)
3.  feature 10 :  FinelineNumber_has_max_by_vn (0.126257)
4.  feature 8  :  Encoded_Upc_nunique_groupby_vn (0.093432)
5.  feature 7  :  FinelineNumber_nunique_groupby_vn (0.086363)
6.  feature 6  :  Encoded_DepartmentDescription_nunique_groupby_vn (0.068791)
7.  feature 11 :  Encoded_DepartmentDescription_max_by_vn (0.065487)
8.  feature 4  :  Encoded_Weekday (0.065123)
9.  feature 3  :  Encoded_Upc (0.060467)
10. feature 2  :  Encoded_DepartmentDescription (0.055747)
11. feature 1  :  FinelineNumber (0.052079)
12. feature 9  :  FinelineNumber_max_by_vn (0.032862)
13. feature 0  :  ScanCount (0.011058)
Score: 0.6123093149686295
Log loss: 2.64528643116
