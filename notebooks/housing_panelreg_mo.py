import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm
    import plotly.express as px
    from typing import List, Dict
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    return Dict, List, PanelOLS, go, make_subplots, np, pd, px, sm


@app.cell
def _(mo):
    mo.md(
        r"""
    # Housing Affordability Panel Regression Project

    ## Overview

    This project demonstrates how to use **panel regression** to analyse housing affordability using synthetic data. The notebook is fully object-oriented, interactive, and leverages Marimo for visualisation and user controls.

    ---

    ## Why Panel Regression?

    **Panel regression** is a statistical method designed for data that follows multiple entities (such as cities or countries) over time. It is especially powerful for:

    - **Controlling for unobserved heterogeneity:** By analysing both cross-sectional (between-entity) and time-series (within-entity) variation, panel regression can account for factors that are constant within an entity but unobserved (e.g., local policies, geography).
    - **Improved causal inference:** By tracking the same entities over time, the method helps isolate the effects of changing variables (like income or unemployment) on outcomes (like house prices or affordability)[2][5][6].

    ---

    ## What is Panel Regression?

    Panel regression models extend classical regression by including both entity and time dimensions. The most common models are:

    - **Fixed Effects (FE):** Controls for all time-invariant differences between entities, focusing on within-entity variation.
    - **Random Effects (RE):** Assumes entity-specific effects are random and uncorrelated with the predictors.
    - **Pooled OLS:** Ignores the panel structure, treating all observations as independent.

    In the context of housing affordability, panel regression allows us to examine how factors like **income** and **unemployment** impact house prices across multiple cities over several years, while controlling for city-specific characteristics[2][5][6].

    ---

    ## How is Panel Regression Used Here?

    ### Data

    - **Entities:** Cities (e.g., Sydney, Melbourne, Brisbane, Perth)
    - **Time:** Years (e.g., 2015–2024)
    - **Variables:**

      - Dependent: House price (proxy for affordability)
      - Independent: Median income, unemployment rate

    ### Process

    1. **Synthetic Data Generation:**  
       The notebook generates synthetic panel data for cities and years, including income, unemployment, and house prices.

    2. **Model Fitting:**  
       A fixed effects panel regression is estimated:
    $$
    \text{House Price}_{it} = \alpha_i + \beta_1 \cdot \text{Income}_{it} + \beta_2 \cdot \text{Unemployment}_{it} + \epsilon_{it}
    $$

    where $i$ indexes cities and $t$ indexes years.

    4. **Interactive Visualisation:**  
       Users can select cities and years to visualise the relationship between variables using Plotly, and see how coefficients change.

    ---

    ## How to Interpret Panel Regression Results (Using This Example)

    - **Coefficients:**  
      - The **income coefficient** ($\beta_1$) shows how much house prices are expected to change, on average, for a one-unit increase in income within a city over time, holding unemployment constant.
      - The **unemployment coefficient** ($\beta_2$) shows the expected change in house prices for a one-unit increase in unemployment, within a city over time.

    - **Example Interpretation:**  
      - If the income coefficient is 1.5, then a $1,000 increase in median income in a city is associated with a $1,500 increase in house prices, all else equal.
      - If the unemployment coefficient is -2, then a 1 percentage point increase in unemployment is associated with a $2,000 decrease in house prices, all else equal.
      - The **fixed effects** control for all unchanging city-specific factors (e.g., geography, persistent local policies), so the model focuses on changes within each city, not differences between cities.

    - **Policy Insight:**  
      Panel regression results can inform policymakers about which factors most strongly drive housing affordability. For example, if unemployment has a strong negative effect, policies to reduce unemployment may also support affordability[2][5][6].

    ---

    ## References

    - [OECD Panel Analysis Example][2]
    - [EU Panel Data Housing Affordability][5]
    - [UK Dynamic Panel Model Example][6]

    ---

    **This project provides a practical, interactive, and extensible template for panel regression analysis of housing affordability, suitable for both technical and policy audiences.**

    [1] https://www.sciencedirect.com/science/article/pii/009411909090011B  
    [2] https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4623269_code2968851.pdf?abstractid=4623269&mirid=1  
    [3] https://www.bis.org/publ/work1149.pdf  
    [4] https://journals.sagepub.com/doi/10.1177/08912424211006178?icid=int.sj-full-text.similar-articles.5  
    [5] https://hrcak.srce.hr/ojs/index.php/crebss/article/download/32777/16874/141989  
    [6] https://eprints.lse.ac.uk/100016/1/Szumilo_housing_affordability.pdf  
    [7] https://www.imf.org/-/media/Files/Publications/WP/2023/English/wpiea2023247-print-pdf.ashx  
    [8] https://www.imf.org/en/Publications/WP/Issues/2023/12/01/Housing-Affordability-A-New-Dataset-541910
    """
    )
    return


@app.cell
def _():
    # Parameter definitions (city-dependent)

    CITIES: list[str] = [
        'Sydney', 'Melbourne', 'Brisbane', 'Perth', 
        'Adelaide', 'Hobart', 'Darwin', 'Canberra'
    ]
    YEARS: list[int] = list(range(2015, 2025))
    SEED: int = 123

    # Median household income estimates (in thousands, i.e. $000s)
    CITY_INCOME_MEAN: dict[str, float] = {
        'Sydney': 120,
        'Melbourne': 110,
        'Brisbane': 105,
        'Perth': 110,
        'Adelaide': 100,
        'Hobart': 95,
        'Darwin': 105,
        'Canberra': 125,
    }
    INCOME_STD: float = 10  # $10,000

    # Unemployment rate bounds (%)
    CITY_UNEMPLOYMENT_LOW: dict[str, float] = {
        'Sydney': 3.5, 'Melbourne': 3.8, 'Brisbane': 3.7, 'Perth': 3.6,
        'Adelaide': 4.0, 'Hobart': 3.9, 'Darwin': 4.2, 'Canberra': 3.5
    }
    CITY_UNEMPLOYMENT_HIGH: dict[str, float] = {
        'Sydney': 5.5, 'Melbourne': 6.0, 'Brisbane': 5.8, 'Perth': 5.7,
        'Adelaide': 6.2, 'Hobart': 5.5, 'Darwin': 6.5, 'Canberra': 5.2
    }

    # Median house prices (in thousands, i.e. $000s) for 2025
    CITY_HOUSE_PRICE_BASE: dict[str, float] = {
        'Sydney': 1700,
        'Melbourne': 1050,
        'Brisbane': 1050,
        'Perth': 950,
        'Adelaide': 1000,
        'Hobart': 800,
        'Darwin': 650,
        'Canberra': 1100,
    }

    # House price sensitivity to income and unemployment
    HOUSE_PRICE_INCOME_COEF: float = 3.0   # Each $1,000 income adds $3,000 to price
    HOUSE_PRICE_UNEMPLOYMENT_COEF: float = -12  # Each 1% unemployment reduces price by $12,000
    HOUSE_PRICE_NOISE_STD: float = 15      # $20,000 noise for realism
    return (
        CITIES,
        CITY_HOUSE_PRICE_BASE,
        CITY_INCOME_MEAN,
        CITY_UNEMPLOYMENT_HIGH,
        CITY_UNEMPLOYMENT_LOW,
        HOUSE_PRICE_INCOME_COEF,
        HOUSE_PRICE_NOISE_STD,
        HOUSE_PRICE_UNEMPLOYMENT_COEF,
        INCOME_STD,
        SEED,
        YEARS,
    )


@app.cell
def _(Dict, List, np, pd):
    class HousingPanelData:
        """
        Generates synthetic panel data for housing affordability analysis,
        with city-dependent parameters and realistic year-on-year house price volatility.

        Attributes:
            cities (List[str]): List of city names.
            years (List[int]): List of years.
            seed (int): Random seed for reproducibility.
            city_income_mean (Dict[str, float]): Mean income per city (in $000s).
            income_std (float): Standard deviation of the income distribution.
            city_unemployment_low (Dict[str, float]): Lower bound for unemployment rate per city (%).
            city_unemployment_high (Dict[str, float]): Upper bound for unemployment rate per city (%).
            city_house_price_base (Dict[str, float]): Base house price per city (in $000s).
            house_price_income_coef (float): Coefficient for income in house price formula.
            house_price_unemployment_coef (float): Coefficient for unemployment in house price formula.
            house_price_noise_std (float): Standard deviation of noise in house price formula.
            house_price_autocorr (float): Autocorrelation factor for house prices (0.7–0.95 recommended).
            df (pd.DataFrame): Generated panel DataFrame with city and year as index.
        """

        def __init__(
            self,
            cities: List[str],
            years: List[int],
            seed: int,
            city_income_mean: Dict[str, float],
            income_std: float,
            city_unemployment_low: Dict[str, float],
            city_unemployment_high: Dict[str, float],
            city_house_price_base: Dict[str, float],
            house_price_income_coef: float,
            house_price_unemployment_coef: float,
            house_price_noise_std: float,
            house_price_autocorr: float = 0.85,
        ) -> None:
            """
            Initializes the data generator with specified (possibly city-dependent) parameters.

            Args:
                cities: List of city names.
                years: List of years.
                seed: Random seed for reproducibility.
                city_income_mean: Dict mapping city to mean income ($000s).
                income_std: Standard deviation of income ($000s).
                city_unemployment_low: Dict mapping city to min unemployment rate (%).
                city_unemployment_high: Dict mapping city to max unemployment rate (%).
                city_house_price_base: Dict mapping city to base house price ($000s).
                house_price_income_coef: Coefficient for income in house price formula.
                house_price_unemployment_coef: Coefficient for unemployment in house price formula.
                house_price_noise_std: Standard deviation of noise in house price formula.
                house_price_autocorr: Autocorrelation factor for house prices (0.7–0.95 recommended).
            """
            self.cities = cities
            self.years = years
            self.seed = seed
            self.city_income_mean = city_income_mean
            self.income_std = income_std
            self.city_unemployment_low = city_unemployment_low
            self.city_unemployment_high = city_unemployment_high
            self.city_house_price_base = city_house_price_base
            self.house_price_income_coef = house_price_income_coef
            self.house_price_unemployment_coef = house_price_unemployment_coef
            self.house_price_noise_std = house_price_noise_std
            self.house_price_autocorr = house_price_autocorr
            self.df = self._generate_data()

        def _generate_data(self) -> pd.DataFrame:
            """
            Generates synthetic panel data with city-dependent parameters and
            realistic, autocorrelated house price evolution.

            Returns:
                pd.DataFrame: Panel DataFrame with city and year as index.
            """
            np.random.seed(self.seed)
            records = []
            for city in self.cities:
                last_price = self.city_house_price_base[city]
                for i, year in enumerate(self.years):
                    income = np.random.normal(self.city_income_mean[city], self.income_std)
                    unemployment = np.random.uniform(
                        self.city_unemployment_low[city],
                        self.city_unemployment_high[city]
                    )
                    # Predicted price from model (without autocorrelation)
                    predicted_price = (
                        self.city_house_price_base[city]
                        + self.house_price_income_coef * income
                        + self.house_price_unemployment_coef * unemployment
                    )
                    # Add autocorrelation: blend previous year's price and new prediction
                    if i == 0:
                        house_price = predicted_price + np.random.normal(0, self.house_price_noise_std)
                    else:
                        house_price = (
                            self.house_price_autocorr * last_price +
                            (1 - self.house_price_autocorr) * predicted_price +
                            np.random.normal(0, self.house_price_noise_std)
                        )
                    last_price = house_price
                    records.append({
                        'city': city,
                        'year': year,
                        'income': income,
                        'unemployment': unemployment,
                        'house_price': house_price
                    })
            df = pd.DataFrame(records)
            return df.set_index(['city', 'year'])

    return (HousingPanelData,)


@app.cell
def _(PanelOLS, pd, sm):
    # Panel Regression model class

    class PanelRegressor:
        """Fits panel regression models to housing affordability data.

        Attributes:
            df (pd.DataFrame): Panel DataFrame with city and year as index.
            results (PanelResults): Fitted regression results (None until fit is called).
        """

        def __init__(self, df: pd.DataFrame) -> None:
            """Initialises the regressor with panel data.

            Args:
                df: Panel DataFrame with city and year as index.
            """
            self.df = df
            self.results = None

        def fit(self) -> PanelOLS:
            """Fits a fixed effects panel regression model.

            Returns:
                PanelOLS: Fitted regression results.
            """
            y = self.df['house_price']
            X = sm.add_constant(self.df[['income', 'unemployment']])
            model = PanelOLS(y, X, entity_effects=True)
            self.results = model.fit()
            return self.results

        def get_coefficients(self) -> pd.Series:
            """Returns the regression coefficients.

            Returns:
                pd.Series: Series of regression coefficients.
            """
            if self.results is None:
                self.fit()
            return self.results.params
    return (PanelRegressor,)


@app.cell
def _(go, make_subplots, pd, px):
    # Visualisation class

    class HousingVisualiser:
        """Visualises housing affordability panel data.

        Attributes:
            df (pd.DataFrame): Panel DataFrame (with reset index).
        """

        def __init__(self, df: pd.DataFrame) -> None:
            """Initialises the visualiser with panel data.

            Args:
                df: Panel DataFrame (with reset index).
            """
            self.df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()

        def scatter_income(self, city: str) -> px.scatter:
            """Creates a scatter plot of income vs house price for a city.

            Args:
                city: Name of the city to filter.

            Returns:
                px.scatter: Plotly scatter plot.
            """
            data = self.df[self.df['city'] == city]
            return px.scatter(
                data, x='income', y='house_price', color='year',
                trendline='ols', title=f'Income vs House Price in {city}'
            )

        def scatter_unemployment(self, city: str) -> px.scatter:
            """Creates a scatter plot of unemployment vs house price for a city.

            Args:
                city: Name of the city to filter.

            Returns:
                px.scatter: Plotly scatter plot.
            """
            data = self.df[self.df['city'] == city]
            return px.scatter(
                data, x='unemployment', y='house_price', color='year',
                trendline='ols', title=f'Unemployment vs House Price in {city}'
            )

        def scatter_year(self, year: int) -> px.scatter:
            """Creates a scatter plot of income vs house price for a year.

            Args:
                year: Year to filter.

            Returns:
                px.scatter: Plotly scatter plot.
            """
            data = self.df[self.df['year'] == year]
            return px.scatter(
                data, x='income', y='house_price', color='city',
                trendline='ols', title=f'Income vs House Price in {year}'
            )

        def income_and_houseprice_vs_year(self, city: str) -> go.Figure:
            """
            Plot income and house price vs year for a single city with dual y-axes.

            Args:
                city (str): The city to plot.

            Returns:
                plotly.graph_objects.Figure: The dual-axis line plot.
            """
            city_data = self.df[self.df['city'] == city]

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Income trace (left y-axis)
            fig.add_trace(
                go.Scatter(
                    x=city_data["year"],
                    y=city_data["income"],
                    name="Income",
                    mode="lines+markers",
                    line=dict(color="royalblue")
                ),
                secondary_y=False,
            )

            # House price trace (right y-axis)
            fig.add_trace(
                go.Scatter(
                    x=city_data["year"],
                    y=city_data["house_price"],
                    name="House Price",
                    mode="lines+markers",
                    line=dict(color="firebrick")
                ),
                secondary_y=True,
            )

            fig.update_layout(
                title_text=f"Income and House Price vs Year in {city}",
                legend=dict(x=0.01, y=0.99),
                xaxis_title="Year",
            )
            fig.update_yaxes(title_text="Income ($'000)", secondary_y=False)
            fig.update_yaxes(title_text="House Price ($'000)", secondary_y=True)

            return fig


        def income_and_houseprice_subplots(self, city: str) -> go.Figure:
            """
            Create a stacked subplot: top panel is dashed income vs year,
            bottom panel is house price vs year, for a single city.

            Args:
                city (str): The city to plot.

            Returns:
                plotly.graph_objects.Figure: The stacked subplot figure.
            """
            city_data = self.df[self.df['city'] == city]

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Median Household Income", "Median House Price")
            )

            # Top panel: Income (dashed line)
            fig.add_trace(
                go.Scatter(
                    x=city_data["year"],
                    y=city_data["income"],
                    name="Income",
                    mode="lines+markers",
                    line=dict(color="royalblue", dash="dash")
                ),
                row=1, col=1
            )

            # Bottom panel: House Price (solid line)
            fig.add_trace(
                go.Scatter(
                    x=city_data["year"],
                    y=city_data["house_price"],
                    name="House Price",
                    mode="lines+markers",
                    line=dict(color="firebrick")
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                title_text=f"Income and House Price vs Year in {city}",
                showlegend=False
            )
            fig.update_yaxes(title_text="Income ($'000)", row=1, col=1)
            fig.update_yaxes(title_text="House Price ($'000)", row=2, col=1)
            fig.update_xaxes(title_text="Year", row=2, col=1)

            return fig

    return (HousingVisualiser,)


@app.cell
def _(
    CITIES: list[str],
    CITY_HOUSE_PRICE_BASE: dict[str, float],
    CITY_INCOME_MEAN: dict[str, float],
    CITY_UNEMPLOYMENT_HIGH: dict[str, float],
    CITY_UNEMPLOYMENT_LOW: dict[str, float],
    HOUSE_PRICE_INCOME_COEF: float,
    HOUSE_PRICE_NOISE_STD: float,
    HOUSE_PRICE_UNEMPLOYMENT_COEF: float,
    HousingPanelData,
    HousingVisualiser,
    INCOME_STD: float,
    PanelRegressor,
    SEED: int,
    YEARS: list[int],
):
    # Generate data, fit model and create chart

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
        house_price_noise_std=HOUSE_PRICE_NOISE_STD
    )


    panel_reg = PanelRegressor(housing_data.df)
    chart = HousingVisualiser(housing_data.df)
    return chart, housing_data, panel_reg


@app.cell
def _(mo):
    mo.md(r"""## Housing Data Visualisation""")
    return


@app.cell
def _(CITIES: list[str], YEARS: list[int], mo):
    # Create UI controls

    city = mo.ui.dropdown(
        options=CITIES, label="Select City for detailed view", value=CITIES[0]
    )
    year = mo.ui.slider(
        start=min(YEARS), stop=max(YEARS), value=min(YEARS), label = "Choose Year"
    )
    return city, year


@app.cell
def _(city, mo, year):
    mo.md(f"""
    {city}
    {year}
    """)
    return


@app.cell
def _(chart, mo, selected_city, selected_year):
    mo.ui.plotly(chart.scatter_income(selected_city))
    mo.ui.plotly(chart.scatter_unemployment(selected_city))
    mo.ui.plotly(chart.scatter_year(selected_year))
    return


@app.cell
def _():
    # mo.ui.plotly(chart.income_and_houseprice_vs_year(selected_city))
    return


@app.cell
def _(chart, mo, selected_city):
    mo.ui.plotly(chart.income_and_houseprice_subplots(selected_city))
    return


@app.cell
def _(city, mo, panel_reg, year):
    # Display results and interactive plots

    selected_city = city.value
    selected_year = year.value
    coefs = panel_reg.get_coefficients()

    mo.md(f"""
    # Housing Affordability Panel Regression

    **Panel regression results:**  
    - Income coefficient: {coefs['income']:.2f}  
    - Unemployment coefficient: {coefs['unemployment']:.2f}  
    - (Intercept): {coefs['const']:.2f}

    """)
    return selected_city, selected_year


@app.cell
def _(housing_data, px):
    fig3d = px.scatter_3d(
        housing_data.df.reset_index(),
        x="income",
        y="unemployment",
        z="house_price",
        color="city",
        animation_frame="year",  # Optional: animate over years
        title="3D Scatter Plot of Raw Housing Data"
    )

    fig3d.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
