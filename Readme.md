# FOODPANDA PAKISTAN

## Customer Segmentation Using K-Means Clustering

**Final Project Report – AI for Business Course**

| **Detail** | **Information** |
|------------|-----------------|
| **Submitted To** | Dr. Altaf Samo |
| **Submitted By** | Tulsi and Aiman |
| **Date** | 6 May 2026 |
| **Dataset** | Foodpanda Pakistan – Kaggle (6,000 records) |
| **Algorithm** | K-Means Clustering (K = 2 to 10 tested) |
| **Optimal K** | K = 3 (selected via Elbow Method + DB Index) |

---

## Executive Summary

This report will present a full customer segmentation project developed on Foodpanda Pakistan, can be developed by any other machine learning algorithm as well, K-means is just one of them. An analysis of 6,000 customer records provided by Kaggle. The main target was to find different groups of customers that could be targeted by different marketing strategies, and thus Foodpanda's very current problem, because they currently apply the same discount for all the customers.

The methodology was designed to be a thorough step by step process involving steps like, Data Verification to provide you with only one row for each customer, Missing Value check on customer clustering features only as no other features are used, Correlation analysis to eliminate redundant features, Standardization and Exercise of various K values from 2 to 4 to be tested. Silhouette Score, Davies-Bouldin Index and the Inertia (Elbow Method) were chosen as the three measures of evaluation. The number of segments to use was determined by both the elbow point, and by business practicality to be K=3.

### Key Findings

| Segment | Size | Avg Order Value | Avg Order Frequency | Churn Rate | Priority Action |
|---------|------|-----------------|---------------------|------------|-----------------|
| Heavy Spenders | 2,216 (37%) | PKR 840.5 | 40.7 orders | 48.5% | VIP re-engagement campaign |
| Casual Browsers | 1,908 (32%) | PKR 772.6 | 16.4 orders | 49.8% | Loyalty onboarding + bundle deals |
| Loyal Engagers | 1,876 (31%) | PKR 781.7 | 16.2 orders | 51.1% | Points activation campaign |

**Conclusion:** Three clear customer segments were found using the K-Means algorithm, which was 40 times brighter than the random baseline (silhouette score: -0.006 baseline vs +0.244 K-Means). These segments offer a clear window into what Foodpanda can learn to do in order to shift from a one-size-measures-all approach to marketing to a precision marketing approach to help increase customer retention, while boosting their revenue at the same time.

---

## Google Colab Link

**Click the link below to access the Google Colab notebook where the code was executed and results were obtained:**

🔗 **[https://colab.research.google.com/drive/1Lyq2cy3hsuMu8cm-5USrWtMlvo9m8Ohd?usp=sharing](https://colab.research.google.com/drive/1Lyq2cy3hsuMu8cm-5USrWtMlvo9m8Ohd?usp=sharing)**

*(This is the link of google colab where I run the code and got results)*

---

## Table of Contents

1. [Title and Problem Statement](#1-title-and-problem-statement)
2. [Business Background and Literature Review](#2-business-background-and-literature-review)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Results and Interpretation](#5-results-and-interpretation)
6. [Business Recommendations](#6-business-recommendations)
7. [Conclusion and Future Implications](#7-conclusion-and-future-implications)
8. [References](#8-references)
9. [Appendix – Complete Python Code](#9-appendix--complete-python-code)

---

## 1. Title and Problem Statement

**Title:** Customer Segmentation using AI for Foodpanda Pakistan's targeted marketing campaign.AI Customer Segmentation for Foodpanda Pakistan's cloud targeted marketing campaign.

### Problem Statement

Foodpanda Pakistan, which is the top online food delivery service in Pakistan, is struggling with its business.In Pakistan, top online food delivery platform Foodpanda is facing a major business challenge. Some restaurant partners in big cities like Karachi, Lahore, Islamabad reacted angrily saying they will temporarily delist and demand the decrease of commission fees between 18-25% instead of 25-35%. The impact of this supply-side problem is that customers have less variety from which to choose among restaurants, and less incentive to stay at the same restaurant.

At present, without understanding the different segments of the customer base, Foodpanda's core business issue is that they follow uniform discounting strategies for all their customers. While this solves issues for one size fits all customers, it washes up marketing dollars for price-insensitive demanders who still make purchases and doesn't offer enough on account of price-sensitive customers who could cancel. Foodpanda cannot go through an efficient allocation of the promotional spend or even go on to negotiate the discounts in the commission structure with the restaurant partners without a data-driven understanding of the segmentations.

### Project Objectives

| Objective Number | Objective Description |
|------------------|------------------------|
| 1 | To examine the Foodpanda Pakistan dataset and verify its suitability for customer segmentation |
| 2 | To apply K-Means clustering algorithm to identify natural groupings within the customer base |
| 3 | To profile each identified segment in plain business language with actionable characteristics |
| 4 | To recommend targeted marketing strategies for each segment that optimize retention and revenue |

### Expected Outcome

The food delivery giant of the country, Foodpanda is currently facing major issues in its business. The partners of restaurants across major cities such as Karachi, Lahore, and Islamabad have raised their concerns and temporarily pulled their restaurants from the platform, calling for the commission fees, which are currently in the range of 25-35%, to be reduced to 18-25%. This scarcity of suppliers will limit restaurant choice from customers and increase the churn rate of customer bases to other businesses.

The heart of the business challenges is the lack of understanding the different categories of customers and implementing uniform discounting strategies for all customers. This approach costs marketers money by sending their marketing dollars to price-insensitive buyers who would buy regardless, and could leave them in a bind if a price-sensitive buyer sees an opportunity and moves on. Without knowing who to target for promotions and what structure to offer for commisions to restaurant partners, Foodpanda can't make efficient use of its budget.

---

## 2. Business Background and Literature Review

### 2.1 Industry Overview – Online Food Delivery in Pakistan

Over the last five years, the online food delivery landscape in Pakistan has undergone tremendous growth on the back of rising smartphone usage, growing internet connectivity, and evolving lifestyle habits of consumers. Competition comes in the form of other applications like Foodpanda, Cheetay and smaller players that exist in the region who have the ability to retain customers is a huge strategic advantage.

### 2.2 The Commission Fee Crisis – A Real Business Problem

The constant problem faced by Foodpanda Pakistan has been to be profitable for them, restaurant partners, and also at a reasonable price for their customers. The restaurant associations in Lahore and Karachi have been asking to reduce the commission over the last few years, going up to 25-35%.In the recent past, both Karachi and Lahore restaurant associations collectively called for a cap on the commission, which currently ranges between 25-35%. As a protest measure, some restaurants temporarily got themselves delisted from the platform.

This supply-side disturbance affects the customer experience, by limiting restaurant choice and extending delivery times and, therefore, inducing customer attrition towards their rivals. But the answer to that can be found in customer segmentation: once they understand the value of each customer segment as well as the cost of acquiring new customers, they can tailor their marketing strategies more specifically, and even offer different commission rates to high-value restaurant partners, to ensure they place bets on the right promotions and other strategies that lead to profitable partnerships.

### 2.3 Customer Segmentation in Food Delivery

The segmentation of customers has been proven to be a best-practice in academic and industrial research in the field of food delivery. Some typical bases are used to segment this industry.

**Customer Lifetime Value (CLV):** Studies have proven that keeping existing customers is much more cost effective than obtaining new customers. A 5% rise in customer retention can boost profits in the food delivery sector by 25-95%. This finding is directly indicative of Business case for Segmentation-based retention strategies.

**Personalization Effectiveness:** Case studies indicate that customer customized offers boost conversions as compared to uniformity by 3-5 times. Transparency earned the trust which led to a 10% increase in orders, and a 48% rise in the Net Promoter Score (NPS) in the "Getting Better Every Day" campaign conducted by Foodpanda itself, after the delays in service.

### 2.4 Theoretical Foundation – K-Means Clustering in Marketing

One of the most popular use of the unsupervised machine learning is K-means clustering in marketing. The method is based on segmenting the data into K clusters, with each data point assigned to the cluster whose centroid (or mean) it's closest to. Advantages of K-Means with regard to customer segmentation are:

| Advantage | Explanation |
|-----------|-------------|
| Interpretability | Cluster centers are averages of features, making them easy to explain to business stakeholders |
| Scalability | K-Means scales linearly with sample size, handling thousands of customers efficiently |
| No labeled data required | As an unsupervised algorithm, it discovers natural groupings without needing pre-labeled training data |
| Proven effectiveness | Extensively validated in academic and industry applications for segmentation |

### 2.5 Gap This Project Addresses

Therefore there are not many studies that have been conducted on customer segmentation in the context of current commission crisis in Foodpanda Pakistan, which is expected to be key to establishing the value of customer segmentation. This project tackles the aforementioned challenge by clustering a recent Foodpanda data set and creating many recommendations tailored to the different segments directly answering the challenges faced by the platform on the supply side.

---

## 3. Dataset Description

### 3.1 Data Source

The data is from Kaggle website, created by Amin Ahmed Khan and November 2025. It is freely accessible and is published. The file name is foodpanda.pk (the all.csv file).

### 3.2 Dataset Size and Structure

| Attribute | Value |
|-----------|-------|
| Total rows | 6,000 |
| Total columns | 20 |
| Data level | One row per customer (verified) |

### 3.3 Column Descriptions

| Column Name | Type | Description |
|-------------|------|-------------|
| customer id | Text | Unique code for each customer (e.g., C5663). One per row – confirms 1 row = 1 customer |
| gender | Category | Male / Female / Other |
| age | Category | Age group: Teenager / Adult / Senior |
| city | Category | Lahore, Karachi, Islamabad, Multan, Peshawar |
| signup date | Date | When the customer first registered on Foodpanda |
| order id | Text | ID of their most recent recorded order |
| order date | Date | Date of their most recent recorded order |
| restaurant name | Text | Restaurant they last ordered from |
| dish name | Text | Specific food item ordered |
| category | Category | Cuisine type: Italian, Fast Food, Chinese, Continental, Dessert |
| quantity | Number | Number of items in the order (1 to 5) |
| **price** | Number | Order value in PKR (100 to 1,500). **Used in clustering** |
| payment method | Category | Cash / Card / Wallet |
| **order frequency** | Number | Total number of times the customer has ordered (1 to 50). **Used in clustering** |
| last order date | Date | When the customer last placed an order |
| **loyalty points** | Number | Total reward points accumulated (0 to 500). **Used in clustering** |
| churned | Category | Active = still orders regularly. Inactive = has stopped ordering |
| rating | Number | Customer satisfaction rating (1 to 5). Has 1,968 missing values. **Not used in clustering** |
| delivery status | Category | Delivered / Delayed / Cancelled |

### 3.4 Clustering Features – Key Statistics

| Feature | Min | Max | Average | Median | Standard Deviation | Missing Values |
|---------|-----|-----|---------|--------|--------------------|----------------|
| price (PKR) | 100 | 1,500 | 801 | 806 | 405 | 0 (None) |
| order frequency | 1 | 50 | 25.3 | 25 | 14.4 | 0 (None) |
| loyalty points | 0 | 500 | 250 | 250 | 145 | 0 (None) |

### 3.5 Data Preprocessing Summary

| Preprocessing Step | Action Taken |
|--------------------|---------------|
| Missing value check | Verified that all three clustering features have zero missing values |
| Data level verification | Confirmed 6,000 unique customers = 6,000 rows = one row per customer |
| Correlation check | Confirmed all feature correlations are near zero (max 0.014), no redundant features |
| Standardization | Applied StandardScaler to ensure all features have mean=0 and variance=1 |
| Outlier treatment | No outliers requiring treatment in clustering features |

**Important Note:** There are 1,968 missing values in the rating column (33%) of the table. No imputation and cleaning was done in this column because it was not used as a clustering feature. That's the only columns that will be used in analysis that will be cleaned with this principle.

---

## 4. Methodology

### 4.1 Overview of Approach

In this project, an unsupervised machine learning technique identified as K-Means algorithm was used for segmentation of Foodpanda customers. The approach adopted eight successive steps starting from the data loading and ending up with visualization. All the steps were performed by means of Python in a Jupyter Notebook environment, utilizing libraries such as pandas, scikit-learn, matplotlib, and seaborn.

### 4.2 Step-by-Step Methodology

**Step 1 – Data Loading and Initial Examination**

The data was put into pandas DataFrame. Initial checks involved certain things: all rows contained a 1-to-1 relationship between rows and unique customer IDs (6,000 rows with 6,000 unique customer IDs); shape (whether there were 6,000 rows and 20 columns, or some other number of rows and columns); the data type of each column; whether there were any records that had a duplicate customer ID; whether there were any records that did not have a customer ID; whether a record was missing a date where it supposed to have one; etc.

**Step 2 – Data Preprocessing (Cleaning)**

Only the three clustering features – price, order frequency, loyalty points – were focused on for data quality problems. The no data entries were also examined and were believed to be zero for all three features. There are no transformations, deletions or imputations on a non-clustering column, including the rating column. The alleged cleaning up was a targeted approach to clean, which was not doing unnecessary work, and was having no impact on data integrity.

**Step 3 – Correlation Check**

Three clustering features are included, for which a correlation matrix has been computed to determine if there are any features strongly correlated with each other. All the correlations were very close to zero, indicating different aspects of customer behaviour in the above features. There were no features that were dropped or amalgamated.

**Step 4 – Feature Standardization**

Three features for clustering (2nd: sampling prevalence, 2nd: IQR and 2nd: Dstn) were standardized using a standard scaling process (Z-score normalization). Each feature was deleted to center around a mean of 0, and to scale down to a standard deviation of 1. The K-Means algorithm is based on Euclidean distance and must be standardized, because otherwise order frequency (1-50) would numerically control what ROWE was standardized on price (100-1500), with regard to importance for the business.

**Step 5 – Naive Baseline**

A baseline with naive classification was set by randomly labelling the 6,000 customers as belonging to any of 3 "clusters" (which simulated K=3) with equal probability 1/3. This random assignment was found to have a silhouette score. This set the minimum achievable standard of performance.

**Step 6 – Finding Optimal K (K=2 to K=10)**

For each value of K from 2 to 10, the k-means clustering algorithm was used. A total of three evaluation metrics have been computed for each K:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| Silhouette Score | Measures how similar a point is to its own cluster versus other clusters | Range -1 to +1; higher is better; >0.5 indicates good separation |
| Davies-Bouldin Index | Measures average similarity between each cluster and its most similar cluster | Lower is better; <1.0 indicates well-separated clusters |
| Inertia (Elbow Method) | Sum of squared distances from each point to its cluster center | Lower is better; optimal K at the "elbow" where drops diminish |

**Step 7 – Final Model Selection (K=3)**

Therefore, based on the elbow method graph (reducing returns after K=3), increase in Davies Bouldin index when K=3 (compared to K=2) and practicality of the business (3 segments manageable by a business marketing team), the optimum number of segments was determined as K=3. A final model for K-Means was tested with K =3.

**Step 8 – Cluster Profiling and Visualization**

For each cluster, each original feature was profiled with the average value for that cluster.Average price, average order frequency, average loyalty points, the size of the clusters and the churn percentage were used to profile each of the three clusters. A three-panel K-selection chart (silhouette, Davies-Bouldin, inertia) and a PCA scatter plot with the three clusters plotted in 2D space were used for visualisation.

### 4.3 Tools and Libraries Used

| Tool/Library | Purpose |
|--------------|---------|
| Python | Programming language |
| pandas | Data manipulation and loading |
| numpy | Numerical operations and random baseline |
| scikit-learn | StandardScaler, KMeans, silhouette_score, davies_bouldin_score, PCA |
| matplotlib | Visualization and plotting |
| seaborn | Enhanced visualizations |
| Google Colab | Execution environment (free, no installation required) |

### 4.4 Justification of Model Choice

K-Means was chosen because it provides interpretable cluster centers, scales efficiently to the dataset size, and produces results that can be easily explained to business decision-makers.

---

## 5. Results and Interpretation

### 5.1 Baseline Performance

| Method | Silhouette Score | Interpretation |
|--------|------------------|----------------|
| Random Baseline (Naive Method) | -0.006 | Essentially zero – meaningless random grouping |
| K-Means (K=3) | +0.244 | 40 times better than baseline; real structure found |

The K-Means model significantly outperformed the random baseline, confirming that the algorithm successfully identified real, learnable patterns in the customer data rather than random noise.

### 5.2 K-Selection Results (K=2 to K=10)

| K | Silhouette Score | Davies-Bouldin Index | Inertia | Verdict |
|---|------------------|----------------------|---------|---------|
| Baseline | -0.006 | N/A | N/A | Random – useless |
| K=2 | 0.244 | 1.640 | 13,477 | Good but only 2 groups – too simple |
| **K=3** | **0.244** | **1.317** | **10,660** | **SELECTED – Elbow point, strong DB improvement** |
| K=4 | 0.270 | 1.131 | 8,505 | Marginal improvement |
| K=5 | 0.272 | 1.045 | 7,091 | Diminishing returns |
| K=6 | 0.287 | 0.977 | 5,947 | Too many segments for practical marketing |
| K=7 | 0.286 | 0.974 | 5,134 | Metrics plateau |
| K=8 | 0.283 | 0.972 | 4,491 | Metrics plateau |
| K=9 | 0.293 | 0.958 | 4,187 | Best raw metrics but 9 segments impractical |
| K=10 | 0.280 | 1.019 | 3,968 | DB worsens; K=9 statistical peak |

**Why K=3 Was Selected:**

1. **Elbow Method:** Inertia dropped sharply from K=2 (13,477) to K=3 (10,660) – a decrease of 2,817. The drop from K=3 to K=4 was only 2,156, indicating diminishing returns beginning at K=3.
2. **Davies-Bouldin Improvement:** DB index improved from 1.640 at K=2 to 1.317 at K=3, a meaningful gain. While higher K values continued improving, gains became progressively smaller.
3. **Business Practicality:** Nine customer segments (K=9) would be impossible for a marketing team to manage simultaneously. Industry best practice recommends 3 to 5 segments for consumer marketing.
4. **Interpretability:** Three segments (high-value heavy users, casual browsers, loyal engagers) map clearly to distinct marketing strategies.

### 5.3 Final Cluster Profiles (K=3)

**Segment 0 – The Heavy Spenders (n = 2,216 customers | 37% of total)**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Average Order Price | PKR 840.5 | Highest spending group of the three segments |
| Average Order Frequency | 40.7 orders | 2.5x more orders than Segments 1 and 2 |
| Average Loyalty Points | 252.6 points | Mid-range loyalty engagement |
| Churn Rate (Inactive %) | 48.5% | Nearly half have already stopped ordering |
| Dominant City | Lahore | Lahore-heavy concentration |
| Top Food Category | Chinese | Prefer Chinese cuisine |
| Preferred Payment | Cash | Not yet using digital payment – missed opportunity |
| Dominant Age Group | Senior | Older customers who order frequently but pay traditionally |

**Business Description:** These are Foodpanda's power users – highest spend per order AND highest ordering frequency. Despite being so active, nearly half have gone inactive, making them the highest-priority re-engagement target. Their cash payment preference suggests they are not fully integrated into the Foodpanda digital ecosystem.

---

**Segment 1 – The Casual Browsers (n = 1,908 customers | 32% of total)**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Average Order Price | PKR 772.6 | Lowest spending – most price-sensitive group |
| Average Order Frequency | 16.4 orders | Infrequent – less than half the frequency of Segment 0 |
| Average Loyalty Points | 115.3 points | Lowest of all segments – barely engaged with rewards |
| Churn Rate (Inactive %) | 49.8% | Almost half are already inactive |
| Dominant City | Multan | Concentrated in Multan |
| Top Food Category | Fast Food | Prefer quick, affordable food |
| Preferred Payment | Card | Digital payment but not the Foodpanda wallet |
| Dominant Age Group | Teenager | Younger, price-conscious demographic |

**Business Description:** Once customers and repeaters, often those who can be drawn by the price of their purchase, who have had little interaction with the loyalty scheme (only 115 points compared to 384 points of the Segment 2). They like to go for cheap and fast. For them, their high low-loyalty engagement indicates that they have yet to make a habit out of frequenting foodpanda.

---

**Segment 2 – The Loyal Engagers (n = 1,876 customers | 31% of total)**

| Metric | Value | What It Means |
|--------|-------|---------------|
| Average Order Price | PKR 781.7 | Mid-range spender |
| Average Order Frequency | 16.2 orders | Similar frequency to Segment 1 – but very different loyalty |
| Average Loyalty Points | 384.6 points | HIGHEST of all segments – deeply engaged with rewards |
| Churn Rate (Inactive %) | 51.1% | Highest churn despite deep loyalty – warning signal |
| Dominant City | Multan | Multan-heavy, like Segment 1 |
| Top Food Category | Italian | Prefer quality, sit-down style cuisines |
| Preferred Payment | Wallet | Fully digital – use Foodpanda's own wallet |
| Dominant Age Group | Teenager | Young, digitally engaged, app-native users |

**Business Description:** According to the findings, these are digitally committed customers: users of the Foodpanda wallet, earned the highest number of loyalty points and like quality cuisines the most. But they have the largest churn rate (51.1%) despite a great degree of platform engagement. This definitely indicates that they are holding on to loyalty points that they haven't redeemed yet and have given up on ordering. This segment could see a big transformation in the form of a points activation campaign.

### 5.4 Key Insights from Results

1. **All three segments show alarmingly high churn rates (48-51%).** This is Foodpanda's most urgent problem – nearly half of all customers across every segment have already stopped ordering.
2. **Segment 0 (Heavy Spenders) drives disproportionate revenue.** With only 37% of customers, they generate significantly higher order frequency (40.7 vs 16 orders for others). Retaining this segment is critical.
3. **Segment 2 (Loyal Engagers) has highest loyalty points but also highest churn.** This paradox indicates that points accumulation alone does not guarantee retention – customers need reminders and incentives to redeem.
4. **City-level differences exist.** Lahore dominates Segment 0; Multan dominates Segments 1 and 2. City-specific campaigns may be necessary.
5. **Payment method correlates with segment.** Wallet users are in Segment 2 (loyal), cash users are in Segment 0 (heavy spenders), card users are in Segment 1 (casual).

---

## 6. Business Recommendations

### 6.1 Segment 0 – Heavy Spenders: VIP Re-Engagement Campaign

| Priority Level | URGENT – 48.5% churn rate on your highest-value customers |
|----------------|-----------------------------------------------------------|
| **Campaign 1** | **Win-Back Campaign:** Send personalized push notifications – "We miss you! Get 30% off your next Chinese order" – targeted at their cuisine preference |
| **Campaign 2** | **VIP Gold Tier:** Create an exclusive membership for top-frequency customers with benefits like free priority delivery, exclusive restaurant access, and monthly vouchers |
| **Campaign 3** | **Cash-to-Wallet Migration:** Offer 100 bonus loyalty points per order paid via Foodpanda Wallet. Since this segment currently pays by cash, migrating them to digital payment increases platform stickiness |
| **Expected Impact** | Recovering even 20% of churned Heavy Spenders (PKR 840 × 40+ orders/year) represents a major revenue recovery opportunity |

### 6.2 Segment 1 – Casual Browsers: Habit-Building & Loyalty Onboarding

| Priority Level | HIGH – Must build ordering habits before they churn permanently |
|----------------|----------------------------------------------------------------|
| **Campaign 1** | **Triple Points Onboarding:** Offer 3x loyalty points on first 5 orders to rapidly bring Casual Browsers into the loyalty ecosystem. Once points accumulate, customers feel invested and are more likely to continue ordering to redeem them |
| **Campaign 2** | **Fast Food Bundle Deals:** Create value combos (e.g., Burger + Fries + Drink for PKR 599) that make the price proposition irresistible for this price-sensitive, fast food-preferring segment |
| **Campaign 3** | **Referral Programme:** Offer PKR 200 credit for every friend referred who places a first order. Young, price-conscious customers respond well to social incentives |
| **Expected Impact** | Converting 25% of this segment from 16 to 25 orders/year across 1,908 customers would add thousands of additional orders annually |

### 6.3 Segment 2 – Loyal Engagers: Reward Activation Campaign

| Priority Level | HIGH – Points expiry warnings can trigger immediate orders |
|----------------|-----------------------------------------------------------|
| **Campaign 1** | **Points Expiry Notifications:** Show customers their current balance and what they can unlock (e.g., "You have 385 points – redeem for a free dessert!"). Many customers accumulate points but forget to use them |
| **Campaign 2** | **Exclusive Italian Restaurant Partnerships:** Partner with premium Italian restaurants in Multan and offer Wallet-exclusive discounts. Directly targets their cuisine preference and digital payment habit |
| **Campaign 3** | **Gamified Monthly Challenges:** Create app challenges (e.g., "Order 5 times this month to unlock a free Italian starter"). This segment's deep app engagement makes them ideal for gamification |
| **Expected Impact** | These customers already trust Foodpanda deeply. Increasing frequency from 16 to 22 orders/year across 1,876 customers would have significant revenue impact at minimal campaign cost |

### 6.4 How These Recommendations Map to the Original Problem

The original business problem was that Foodpanda applies uniform discounts across all customers without understanding segments. These recommendations directly solve that problem by:

| Original Problem | Solution from Segmentation |
|------------------|---------------------------|
| Uniform 30% discount given to everyone | Targeted offers: Heavy Spenders get VIP benefits, Casual Browsers get bundle deals, Loyal Engagers get points activation |
| Marketing budget wasted on price-insensitive customers | Heavy Spenders receive loyalty perks (not discounts that would have happened anyway) |
| High churn rates across all customers | Segment-specific retention campaigns addressing each group's unique reason for churning |
| Restaurant commission protests | Differential commission negotiations possible: offer lower commissions for restaurants frequented by Casual Browsers (price-sensitive) while justifying standard commissions for restaurants popular with Heavy Spenders |

---

## 7. Conclusion and Future Implications

### 7.1 Summary of Findings

A successful application of K-Means Clustering to segment Pakistan's customer base into three segments: Heavy Spenders (37% of the users, order frequency is high, the most expensive side of the users base), Casual Browsers (32% of the users, more price sensitive, low LTV), and Loyal Engagers (31% of the users, most engaged politically, high churn). The K-Means model performed significantly better than the random baseline, with a performance improvement of 40 times, which further supports the fact that K-Means has identified a true learnable pattern in the customer data.

### 7.2 Limitations of This Study

| Limitation | Explanation | Impact |
|------------|-------------|--------|
| Synthetic dataset | The Kaggle dataset appears computer-generated with uniform distributions. Real Foodpanda transaction data would produce sharper, more actionable segments | Clusters may be less distinct than in real data |
| Single time snapshot | Analysis reflects customer behavior at one moment. Seasonal trends (Ramadan, exam season, summer heat) affect ordering behavior significantly | Recommendations may need seasonal adjustment |
| Three features only | Including additional signals like delivery success rate, time-of-day ordering, or cuisine diversity might reveal richer segments | Some customer nuances may be missed |
| Uniform churn across segments | All segments show ~50% churn. This uniformity is atypical of real data and reflects the synthetic nature of the dataset | Churn predictions may not generalize to real Foodpanda data |

### 7.3 Future Implications and Extensions

**For Foodpanda Management:**

1. **Implement real-time segmentation:** The clustering model can then be used every week on new customer data to gain insight into the changes in segments over time.
2. **Do A/B tests for the recommendations:** Foodpanda should do some controlled small experiments that involve delivering the VIP campaign to 10% of Heavy Spenders (say), and then comparing that arm with the control arm to detect what the actual lift is before going all out.
3. **Use more data sources:** Include information about delivery time, customer support interactions and promotions that are redeemed.

### 7.4 Final Conclusion

In this project, we've been successful in applying K Means Clustering to the customer base of the Pakistani food delivery app called Foodpanda to draw three solid separate groups that we can take action against. The analysis was carried out on a rigorous methodology which involved data verification, correlation check, baseline pre-test, and testing of K- values of 2 to 10. From business practicality, elbow method, Davies-Bouldin method, it is decided that K=3 is the optimum number of segments.

The segments are clearly identified and distinct each with its own behavioral profile and marketing needs – Heavy Spenders, Casual Browsers and loyal Engagers. The specific recommendation for win-back, loyalty initiation and points activation are specific to Foodpanda's existing business challenge on uniform discounting and high churn rates.

There is a big lesson in this project that machine learning can turn raw customer data into valuable business decisions without on-the-job statistics education. With proper explanation, implemented methods such as K-Means can help companies such as Foodpanda shift from one size fits all marketing to precision marketing, and achieve both enhanced customer retention and increased revenue.

---

## 8. References

| Reference | Source |
|-----------|--------|
| Foodpanda Pakistan Dataset | Kaggle – Amin Ahmed Khan |
| K-Means Clustering Algorithm | Scikit-learn documentation |
| Silhouette Score | Rousseeuw (1987) |
| Davies-Bouldin Index | Davies & Bouldin (1979) |
| Customer Segmentation in Food Delivery | Chen et al. (2021) |
| Foodpanda Business News | Profit Magazine Pakistan |
| AI Assistant for Technical Work | Claude AI (Anthropic, 2026) |

---

## 9. Appendix – Complete Python Code and Step-by-Step Instructions

Below is the complete Python script used for this analysis. To run it:

1. Go to **Google Colab** ([colab.research.google.com](https://colab.research.google.com)) – free, no installation needed
2. Upload your Foodpanda Excel/CSV file
3. Copy and paste the code below into a new notebook cell
4. Change `'your_file.xlsx'` to your actual filename
5. Run each cell step by step

### Step-by-Step Instructions How to Run the Data:

| Step | What to Do |
|------|------------|
| 1 | Go to [colab.research.google.com](https://colab.research.google.com) and sign in with your Google account |
| 2 | Click "New Notebook" |
| 3 | Click the "Upload" button and upload your Foodpanda Excel file |
| 4 | Click "+ Code" to create a new code cell |
| 5 | Copy the Python code from this appendix |
| 6 | Paste it into the code cell |
| 7 | Change `'your_file.xlsx'` to your actual filename |
| 8 | Click the play button (▶) or press Shift + Enter |

### Complete Python Code

```python
# ── STEP 1: Import all libraries ────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── STEP 2: Load the data ────────────────────────────────────────────

# Replace 'your_file.xlsx' with your actual file name
df = pd.read_excel('your_file.xlsx')
print('Shape:', df.shape)  # Should be (6000, 20)
print('Unique customers:', df['customer id'].nunique())  # Should be 6000

# ── STEP 3: Verify data level (1 row = 1 customer) ──────────────────

assert df['customer id'].nunique() == len(df), 'Not one row per customer!'
print('Data level confirmed: 1 row per customer')

# ── STEP 4: Check missing values in CLUSTERING FEATURES ONLY ────────

features = ['price', 'order frequency', 'loyalty points']
print(df[features].isnull().sum())  # Should all be 0

# ── STEP 5: Correlation check ────────────────────────────────────────

print(df[features].corr().round(3))  # All near zero = good, keep all features

# ── STEP 6: Standardize features ─────────────────────────────────────

X_raw = df[features].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── STEP 7: Naive baseline ───────────────────────────────────────────

np.random.seed(42)
baseline = np.random.randint(0, 3, size=len(X))
print(f'Baseline Silhouette: {silhouette_score(X, baseline):.4f}') # ~ -0.006

# ── STEP 8: Test K=2 to K=10 ─────────────────────────────────────────

sil, db, inertia = [], [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil.append(silhouette_score(X, labels))
    db.append(davies_bouldin_score(X, labels))
    inertia.append(km.inertia_)
    print(f'K={k} Sil={sil[-1]:.4f} DB={db[-1]:.4f} Inertia={inertia[-1]:.1f}')

# ── STEP 9: Plot the 3 metrics ───────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(K_range, sil, marker='o', color='blue')
axes[0].axhline(y=0.244, color='red', linestyle='--', label='K=3 value')
axes[0].set_title('Silhouette Score (Higher is Better)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Silhouette Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, db, marker='o', color='green')
axes[1].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('DB Index')
axes[1].grid(True, alpha=0.3)

axes[2].plot(K_range, inertia, marker='o', color='red')
axes[2].axvline(x=3, color='orange', linestyle='--', label='Elbow at K=3')
axes[2].set_title('Inertia / Elbow Method')
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Inertia')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── STEP 10: Apply final model K=3 ──────────────────────────────────

best_k = 3  # Selected based on elbow + DB improvement + business rationale
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X)

# ── STEP 11: Describe clusters (using ORIGINAL unscaled values) ──────

profile = df.groupby('Cluster').agg(
    avg_price=('price', 'mean'),
    avg_order_freq=('order frequency', 'mean'),
    avg_loyalty=('loyalty points', 'mean'),
    count=('customer id', 'count'),
    churn_pct=('churned', lambda x: (x == 'Inactive').mean() * 100)
).round(1)

profile['percentage'] = (profile['count'] / len(df) * 100).round(1)
profile = profile[['count', 'percentage', 'avg_price', 'avg_order_freq', 'avg_loyalty', 'churn_pct']]
profile.columns = ['Count', 'Percentage', 'Avg Price (PKR)', 'Avg Order Frequency', 'Avg Loyalty Points', 'Churn %']
print(profile)

# ── STEP 12: PCA visualization ───────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

colors = ['red', 'blue', 'green']
cluster_names = {0: 'Heavy Spenders', 1: 'Casual Browsers', 2: 'Loyal Engagers'}

plt.figure(figsize=(10, 6))
for c in range(best_k):
    mask = df['Cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[c], alpha=0.5, s=15, 
                label=f'{cluster_names[c]} (Cluster {c})')

plt.title('Foodpanda Customer Segments - PCA Visualization')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ── STEP 13: Baseline comparison bar chart ──────────────────────────

baseline_sil = silhouette_score(X, baseline)
kmeans_sil = silhouette_score(X, df['Cluster'])

plt.figure(figsize=(6, 4))
bars = plt.bar(['Random Baseline (Naive)', 'K-Means (K=3)'], 
               [baseline_sil, kmeans_sil], 
               color=['gray', 'green'])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title('K-Means Outperforms Random Baseline by 40x')
plt.ylabel('Silhouette Score (Higher is Better)')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom')
plt.show()

print(f"\nFinal Summary:")
print(f"Baseline Silhouette Score (Random): {baseline_sil:.4f}")
print(f"K-Means Silhouette Score (K=3): {kmeans_sil:.4f}")
print(f"Improvement Factor: {kmeans_sil / baseline_sil:.0f}x better than baseline")
