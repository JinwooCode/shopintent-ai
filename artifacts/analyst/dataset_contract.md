# ShopIntent AI · Dataset contract

_Machine-checked part: `dataset_contract.json`, built by code and enforced by the validation tools. Column documentation written by the Contract Author agent (gpt-4.1-mini) (6201168a5ff7)._

This dataset contains 12,330 completed user sessions with 15.5% resulting in purchases. Each row represents a distinct session with no missing values and includes features about page views, durations, and anonymized categorical codes. Use this contract to understand each column's meaning and modeling guidance, noting that PageValues is excluded from the main model and Revenue is the prediction target.

**Rows:** 12,330 · **Target:** `Revenue` (positive rate 15.5%) · **Excluded from the main model:** `PageValues`

## Columns

| Column | Type | Allowed values / range | Meaning | How to use |
|---|---|---|---|---|
| `Administrative` | integer | 0 to ∞ | Number of account-management pages the visitor viewed in the session. | Use as a feature representing user interaction with account-management pages; integer values >= 0. |
| `Administrative_Duration` | float | 0 to ∞ | Total seconds spent on account-management pages in the session. | Use as a feature representing time spent on account-management pages; float values >= 0. |
| `Informational` | integer | 0 to ∞ | Number of informational pages (about the site, contact, policies) viewed in the session. | Use as a feature representing user interaction with informational pages; integer values >= 0. |
| `Informational_Duration` | float | 0 to ∞ | Total seconds spent on informational pages in the session. | Use as a feature representing time spent on informational pages; float values >= 0. |
| `ProductRelated` | integer | 0 to ∞ | Number of product pages viewed in the session. | Use as a feature representing user interest in products; integer values >= 0. |
| `ProductRelated_Duration` | float | 0 to ∞ | Total seconds spent on product pages in the session. | Use as a feature representing time spent on product pages; float values >= 0. |
| `BounceRates` | float | 0 to 1 | Average bounce rate (Google Analytics) of the pages viewed: share of visitors who entered on that page and left without another request. | Use as a feature representing page bounce behavior; float values within [0, 1]. |
| `ExitRates` | float | 0 to 1 | Average exit rate (Google Analytics) of the pages viewed: share of page views that were the last of their session. | Use as a feature representing page exit behavior; float values within [0, 1]. |
| `PageValues` | float | 0 to ∞ | Average Google Analytics page value of the pages viewed before a completed transaction. | Exclude from the main model as it strongly leaks the purchase outcome. |
| `SpecialDay` | float | 0 to 1 | Closeness of the visit date to a special day such as Mother's Day or Valentine's Day, from 0 (not close) to 1 (closest). | Use as a feature indicating temporal proximity to special days; float values within [0, 1]. |
| `Month` | category | Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec | Month of the visit. | Use as a categorical feature with allowed values: Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec. |
| `OperatingSystems` | category_code | 1, 2, 3, 4, 5, 6, 7, 8 | Anonymized code of the visitor's operating system. | Use as a categorical feature with allowed codes 1 to 8; these are category codes, not quantities. |
| `Browser` | category_code | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | Anonymized code of the visitor's browser. | Use as a categorical feature with allowed codes 1 to 13; these are category codes, not quantities. |
| `Region` | category_code | 1, 2, 3, 4, 5, 6, 7, 8, 9 | Anonymized code of the visitor's region. | Use as a categorical feature with allowed codes 1 to 9; these are category codes, not quantities. |
| `TrafficType` | category_code | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | Anonymized code of the traffic source that brought the visitor (e.g. direct, banner, SMS). | Use as a categorical feature with allowed codes 1 to 20; these are category codes, not quantities. |
| `VisitorType` | category | New_Visitor, Other, Returning_Visitor | Whether the visitor is new, returning or other. | Use as a categorical feature with allowed values: New_Visitor, Other, Returning_Visitor. |
| `Weekend` | boolean | False, True | Whether the visit happened on a weekend. | Use as a boolean feature with values False or True. |
| `Revenue` | boolean | False, True | Whether the session ended in a purchase. | Use as the prediction target; boolean values False or True. |

## Assumptions

- Each row is one completed session; each session belongs to a different user.
- Sessions are complete, so predictions are a visit simulation, not real-time scoring.
- Identical rows are distinct single-page bounce sessions and are kept.
- OperatingSystems, Browser, Region and TrafficType are anonymized category codes, not quantities.

## Constraints (enforced)

- No missing values in any column.
- Counts and durations are >= 0.
- BounceRates, ExitRates and SpecialDay are within [0, 1].
- Categorical columns contain only the listed allowed_values.
- PageValues is strongly tied to the purchase outcome; the main model must not use it.
