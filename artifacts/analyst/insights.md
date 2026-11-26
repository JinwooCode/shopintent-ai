# ShopIntent AI · Business insights

_Written by the Insights Analyst agent (gpt-4.1-mini) from the verified facts in `eda_stats.json` (stats 06310c9d2ea8). Charts and tables: `eda_report.html`._

## Summary

The store's purchase rates vary significantly by month, with November showing the highest rate at 25.4%. New visitors have a notably higher purchase rate (24.9%) compared to returning visitors (13.9%). Visitors who spend more time and view more product pages are more likely to buy. Lower bounce and exit rates correlate with higher purchase likelihood. Traffic source 2 yields the highest purchase rate at 21.6%.

## Insights

### 1. November Shows Peak Purchase Rate

The purchase rate in November is 25.4% (2,998 sessions, 760 buyers), the highest among all months, while February has the lowest at 1.6% (184 sessions, 3 buyers).

**What the store can do:** Focus marketing campaigns and inventory planning around November to capitalize on higher purchase rates, and investigate factors contributing to low February sales for improvement.

### 2. New Visitors Purchase More Frequently Than Returning Ones

New visitors have a purchase rate of 24.9% (1,694 sessions, 422 buyers), which is higher than returning visitors at 13.9% (10,551 sessions, 1,470 buyers).

**What the store can do:** Enhance first-time visitor experience and onboarding to maintain high purchase rates and convert new visitors into loyal customers.

### 3. Higher Engagement on Product Pages Links to Purchases

Buyers view a median of 29.0 product pages and spend 18.5 minutes on product pages, compared to non-buyers who view 16.0 pages and spend 8.5 minutes. Correlations of ProductRelated (0.159) and ProductRelated_Duration (0.152) with purchase support this.

**What the store can do:** Improve product page content and navigation to encourage longer browsing and more page views, potentially increasing purchase likelihood.

### 4. Lower Bounce and Exit Rates Associate with Buying

Buyers have a median bounce rate of 0.0% and exit rate of 1.6%, while non-buyers have 0.4% bounce and 2.9% exit rates. Correlations of ExitRates (-0.207) and BounceRates (-0.151) with purchase are negative.

**What the store can do:** Optimize landing pages and site flow to reduce bounce and exit rates, aiming to keep visitors engaged and increase purchases.

### 5. Traffic Source 2 Drives Highest Purchase Rate

Traffic source 2 has a purchase rate of 21.6% (3,913 sessions, 847 buyers), outperforming other sources such as source 1 at 10.7% and source 3 at 8.8%.

**What the store can do:** Allocate more marketing resources to traffic source 2 and analyze its characteristics to replicate its success across other channels.

### 6. Weekend Visits Yield Higher Purchase Rates

Purchase rate on weekends is 17.4% (2,868 sessions, 499 buyers), higher than weekdays at 14.9% (9,462 sessions, 1,409 buyers).

**What the store can do:** Consider targeted promotions or campaigns during weekends to leverage higher purchase propensity.

## Caveats

- Purchase rates for months with fewer sessions (e.g., February with 184 sessions) may be less stable.
- Traffic source 13 has a small sample size (738 sessions), so its purchase rate should be interpreted cautiously.
- PageValues is left out of these insights and of the main model: it is computed from pages that led to a purchase, so it leaks the answer.
- These are associations in one store's data over one year, not proof of cause and effect.
