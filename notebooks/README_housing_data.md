# Synthetic Australian Housing Panel Data

## Overview

This synthetic dataset simulates Australian capital city housing markets for use in panel regression, visualisation, and policy analysis. It is generated using a custom Python class that models key economic variables for each city and year, with realistic trends and volatility.

---

## Data Structure

- **Entities:** Australian capital cities (`city`)
- **Time:** Annual observations, 2015–2024 (`year`)
- **Variables:**
  - `income`: Median annual household income for the city (in thousands of dollars, e.g., `120` = $120,000)
  - `unemployment`: City unemployment rate (%) (e.g., `4.3` = 4.3%)
  - `house_price`: Median house price for the city (in thousands of dollars, e.g., `1700` = $1,700,000)

Each row represents a unique (city, year) combination.

---

## Example Data

| city      | year | income | unemployment | house_price |
|-----------|------|--------|--------------|-------------|
| Sydney    | 2015 | 118.3  | 4.1          | 1,720.5     |
| Sydney    | 2016 | 121.2  | 3.9          | 1,765.2     |
| Melbourne | 2015 | 109.8  | 4.6          | 1,041.7     |
| ...       | ...  | ...    | ...          | ...         |

---

## Parameter Definitions

All parameters are set to reflect (potentially) typical values for Australian cities in the 2015–2025 period.
However these values have not been validated against real data and are purely illustrative.

| Parameter                       | Description                                           | Example (Sydney) |
|----------------------------------|-------------------------------------------------------|------------------|
| `city_income_mean`               | Mean household income per city ($000s)                | 120              |
| `income_std`                     | Std dev of income ($000s)                             | 20               |
| `city_unemployment_low/high`     | Min/max unemployment rate per city (%)                | 3.5 / 5.5        |
| `city_house_price_base`          | Base house price per city ($000s)                     | 1700             |
| `house_price_income_coef`        | \$1,000 income → house price increase (\$)              | 3.0              |
| `house_price_unemployment_coef`  | 1% unemployment → house price change (\$)              | -12              |
| `house_price_noise_std`          | Std dev of annual price noise (\$000s)                 | 20               |
| `house_price_autocorr`           | Autocorrelation (smooths year-to-year price changes)  | 0.85             |

---

## How the Data is Generated

For each city and year:
- **Income** is sampled from a normal distribution with city-specific mean and standard deviation.
- **Unemployment** is sampled from a uniform distribution within city-specific bounds.
- **House price** is calculated as:

$$
\text{house\_price}_{city, year} = \text{base}_{city} + \beta_1 \cdot \text{income} + \beta_2 \cdot \text{unemployment}
$$

To mimic real markets, **autocorrelation** is applied so each year’s price is mostly based on the previous year’s price, with only small random and model-driven changes. This keeps year-on-year volatility realistic (typically 3–8%).

---

## Why Use This Synthetic Data?

- **Privacy:** No real household or property data is used.
- **Realism:** Parameter choices and autocorrelation reflect real Australian market behavior.
- **Flexibility:** Easily adjust parameters to simulate shocks, booms, or busts.
- **Panel Structure:** Suitable for econometric analysis, time series, and machine learning.

---

## Example Generation Code

```python
housing_data = HousingPanelData(
    cities=CITIES,
    years=YEARS,
    seed=SEED,
    city_income_mean=CITY_INCOME_MEAN,
    income_std=INCOME_STD,
    city_unemployment_low=CITY_UNEMPLOYMENT_LOW,
    city_unemployment_high=CITY_UNEMPLOYMENT_HIGH,
    city_house_price_base=CITY_HOUSE_PRICE_BASE,
    house_price_income_coef=HOUSE_PRICE_INCOME_COEF,
    house_price_unemployment_coef=HOUSE_PRICE_UNEMPLOYMENT_COEF,
    house_price_noise_std=HOUSE_PRICE_NOISE_STD,
    house_price_autocorr=0.85,
)
df = housing_data.df
```

---

## Typical Use Cases

- **Panel regression analysis** (e.g., fixed effects models)
- **Time series visualisation** (income and house prices over time)
- **Policy simulation and scenario analysis**
- **Teaching and demonstration** (safe, reproducible, and realistic)