"""
Spread Analysis
===============
Analysis of inter-commodity and crack spreads.
"""

from datetime import datetime

import pandas as pd
from scipy import stats


class SpreadAnalyzer:
    """
    Spread analysis for oil markets.

    Features:
    - WTI-Brent spread analysis
    - Crack spread calculations (3-2-1, 2-1-1, etc.)
    - Regional differentials
    - Historical percentile rankings

    Unit Reference:
    - Crude oil (WTI, Brent): Quoted in $/barrel
    - RBOB Gasoline: Quoted in $/gallon (42 gallons per barrel)
    - Heating Oil: Quoted in $/gallon (42 gallons per barrel)
    - Gasoil (ICE): Quoted in $/metric tonne (~7.45 barrels per tonne)
    """

    # Standard conversion: gallons per barrel
    GALLONS_PER_BARREL = 42

    # Approximate barrels per metric tonne for gasoil
    BARRELS_PER_TONNE_GASOIL = 7.45

    # Contract specifications for spread calculations
    # Maps Bloomberg ticker prefix to conversion factor to get $/barrel
    BARREL_CONVERSION = {
        "CL": 1,      # WTI - already in $/bbl
        "CO": 1,      # Brent - already in $/bbl
        "XB": 42,     # RBOB - $/gal to $/bbl (42 gal/bbl)
        "HO": 42,     # Heating Oil - $/gal to $/bbl
        "QS": 7.45,   # Gasoil - $/tonne to $/bbl (approx)
    }

    # Regional refining margin configurations (simplified).
    REGIONAL_REFINING_CONFIGS = {
        "usgc_321": {
            "display_name": "US Gulf Coast 3-2-1",
            "region": "US Gulf Coast",
            "crude": {"key": "wti", "label": "WTI (CL1)", "unit": "barrel", "ratio": 3},
            "products": [
                {"key": "gasoline", "label": "RBOB Gasoline (XB1)", "unit": "gallon", "ratio": 2},
                {"key": "distillate", "label": "NY ULSD (HO1)", "unit": "gallon", "ratio": 1},
            ],
            "formula": "((2 × Gasoline + 1 × Distillate) − 3 × WTI) ÷ 3",
            "source": "U.S. EIA '3:2:1 Crack Spread' explainer (2013)",
            "source_url": "https://www.eia.gov/todayinenergy/includes/crackspread_explain.php",
            "notes": "Replicates the standard Gulf Coast 3-2-1 crack spread definition.",
        },
        "nwe_312": {
            "display_name": "Northwest Europe 3-1-2",
            "region": "Northwest Europe",
            "crude": {"key": "brent", "label": "Brent (CO1)", "unit": "barrel", "ratio": 3},
            "products": [
                {"key": "gasoline", "label": "Eurobob proxy (RBOB)", "unit": "gallon", "ratio": 1},
                {"key": "gasoil", "label": "ICE Gasoil (QS1)", "unit": "tonne", "ratio": 2},
            ],
            "formula": "((1 × Gasoline + 2 × Gasoil) − 3 × Brent) ÷ 3",
            "source": "IEA Global Indicator Refinery Margins Methodology (Aug 2024)",
            "source_url": "https://iea.blob.core.windows.net/assets/b6542edd-41b7-4a11-988f-4ffbc0a28e3b/IEARefineryMarginMethodologyAugust2024.pdf",
            "notes": "Weights gasoline vs distillate output per the diesel-heavy NWE yield slate outlined by the IEA.",
        },
        "singapore_complex": {
            "display_name": "Singapore Complex Margin",
            "region": "Singapore",
            "crude": {"key": "dubai", "label": "Dubai Swap (DAT2)", "unit": "barrel", "ratio": 3},
            "products": [
                {"key": "gasoline", "label": "Mogas proxy (RBOB)", "unit": "gallon", "ratio": 1.4},
                {"key": "gasoil", "label": "Gasoil / Jet proxy (QS1)", "unit": "tonne", "ratio": 1.6},
            ],
            "formula": "((1.4 × Mogas + 1.6 × Gasoil) − 3 × Dubai) ÷ 3",
            "source": "IEA Global Indicator Refinery Margins Methodology (Aug 2024)",
            "source_url": "https://iea.blob.core.windows.net/assets/b6542edd-41b7-4a11-988f-4ffbc0a28e3b/IEARefineryMarginMethodologyAugust2024.pdf",
            "notes": "Approximates the Singapore complex slate by grouping light products and middle distillates; heavier fuel oil components are omitted when live benchmarks are unavailable.",
        },
    }

    def __init__(self):
        """Initialize spread analyzer."""
        pass

    @classmethod
    def convert_to_barrel(cls, price: float, unit: str) -> float:
        """
        Convert a price to $/barrel.

        Args:
            price: Price in the given unit
            unit: Unit of the price - 'gallon', 'barrel', or 'tonne'

        Returns:
            Price in $/barrel
        """
        if unit == 'barrel':
            return price
        elif unit == 'gallon':
            return price * cls.GALLONS_PER_BARREL
        elif unit == 'tonne':
            return price * cls.BARRELS_PER_TONNE_GASOIL
        else:
            raise ValueError(f"Unknown unit: {unit}. Expected 'gallon', 'barrel', or 'tonne'")

    def calculate_wti_brent_spread(
        self,
        wti_price: float,
        brent_price: float
    ) -> dict:
        """
        Analyze WTI-Brent spread.

        Args:
            wti_price: WTI price
            brent_price: Brent price

        Returns:
            Spread analysis
        """
        spread = wti_price - brent_price
        spread_pct = (spread / brent_price) * 100

        # Historical context (typical range -8 to +2)
        if spread > 0:
            direction = "WTI Premium"
        elif spread < -5:
            direction = "Wide Brent Premium"
        else:
            direction = "Normal Brent Premium"

        return {
            "spread": round(spread, 2),
            "spread_pct": round(spread_pct, 2),
            "wti_price": round(wti_price, 2),
            "brent_price": round(brent_price, 2),
            "direction": direction,
        }

    def calculate_crack_spread(
        self,
        crude_price: float,
        gasoline_price: float,
        distillate_price: float,
        crack_type: str = "3-2-1",
        gasoline_unit: str = "gallon",
        distillate_unit: str = "gallon"
    ) -> dict:
        """
        Calculate crack spread.

        The crack spread represents the refining margin - the difference between
        the value of refined products and the cost of crude oil input.

        Standard unit conventions:
        - Crude oil (WTI/Brent): Always quoted in $/barrel
        - RBOB Gasoline: Quoted in $/gallon (multiply by 42 to get $/barrel)
        - Heating Oil: Quoted in $/gallon (multiply by 42 to get $/barrel)

        Args:
            crude_price: Crude oil price in $/barrel
            gasoline_price: Gasoline price (default: $/gallon from Bloomberg XB1)
            distillate_price: Distillate/heating oil price (default: $/gallon from Bloomberg HO1)
            crack_type: Type of crack spread ('3-2-1', '2-1-1', 'gasoline', 'heating_oil')
            gasoline_unit: Unit of gasoline price - 'gallon' or 'barrel' (default: 'gallon')
            distillate_unit: Unit of distillate price - 'gallon' or 'barrel' (default: 'gallon')

        Returns:
            Dict containing crack spread analysis with keys:
            - crack_spread: The crack spread value in $/barrel
            - crack_type: Type of crack spread calculated
            - description: Human-readable description
            - crude_price: Input crude price
            - gasoline_bbl: Gasoline price converted to $/barrel
            - distillate_bbl: Distillate price converted to $/barrel
            - margin_pct: Refining margin as percentage of crude price
        """
        # Convert product prices to $/barrel using explicit units
        gasoline_bbl = self.convert_to_barrel(gasoline_price, gasoline_unit)
        distillate_bbl = self.convert_to_barrel(distillate_price, distillate_unit)

        # Calculate crack based on type
        if crack_type == "3-2-1":
            # 3-2-1: 3 barrels crude -> 2 barrels gasoline + 1 barrel distillate
            crack = (2 * gasoline_bbl + distillate_bbl - 3 * crude_price) / 3
            description = "3-2-1 (2 gas + 1 dist - 3 crude)"

        elif crack_type == "2-1-1":
            # 2-1-1: 2 barrels crude -> 1 barrel gasoline + 1 barrel distillate
            crack = (gasoline_bbl + distillate_bbl - 2 * crude_price) / 2
            description = "2-1-1 (1 gas + 1 dist - 2 crude)"

        elif crack_type == "gasoline":
            # Simple gasoline crack
            crack = gasoline_bbl - crude_price
            description = "Gasoline Crack (gas - crude)"

        elif crack_type == "heating_oil":
            # Simple heating oil crack
            crack = distillate_bbl - crude_price
            description = "Heating Oil Crack (HO - crude)"

        else:
            crack = (2 * gasoline_bbl + distillate_bbl - 3 * crude_price) / 3
            description = "3-2-1 Default"

        # Refining margin percentage
        margin_pct = (crack / crude_price) * 100

        return {
            "crack_spread": round(crack, 2),
            "crack_type": crack_type,
            "description": description,
            "crude_price": round(crude_price, 2),
            "gasoline_bbl": round(gasoline_bbl, 2),
            "distillate_bbl": round(distillate_bbl, 2),
            "margin_pct": round(margin_pct, 2),
        }

    def calculate_regional_refining_margin(
        self,
        region_key: str,
        prices: dict[str, float]
    ) -> dict | None:
        """
        Calculate a simplified regional refining margin.

        Args:
            region_key: Configuration key (see REGIONAL_REFINING_CONFIGS)
            prices: Mapping of price inputs keyed by commodity (e.g. 'wti', 'gasoline')

        Returns:
            Dict of refining margin metrics, or None when required prices are missing.
        """
        config = self.REGIONAL_REFINING_CONFIGS.get(region_key)
        if not config:
            raise ValueError(f"Unknown refining margin region: {region_key}")

        crude_def = config["crude"]
        crude_price = prices.get(crude_def["key"])
        if crude_price is None:
            return None

        crude_ratio = crude_def["ratio"]
        crude_price_bbl = self.convert_to_barrel(crude_price, crude_def["unit"])
        crude_cost = crude_price_bbl * crude_ratio

        product_contributions = []
        total_product_value = 0.0

        for product in config["products"]:
            price = prices.get(product["key"])
            if price is None:
                return None
            price_bbl = self.convert_to_barrel(price, product["unit"])
            contribution = price_bbl * product["ratio"]
            total_product_value += contribution
            product_contributions.append(
                {
                    "label": product["label"],
                    "ratio": product["ratio"],
                    "price_per_bbl": round(price_bbl, 2),
                    "price_per_bbl_raw": price_bbl,
                    "value_component": round(contribution, 2),
                    "value_component_raw": contribution,
                    "unit": product["unit"],
                }
            )

        margin = (total_product_value - crude_cost) / crude_ratio
        product_value_per_bbl = total_product_value / crude_ratio
        margin_pct = (margin / crude_price_bbl * 100) if crude_price_bbl else 0.0

        return {
            "region_key": region_key,
            "display_name": config["display_name"],
            "region": config["region"],
            "margin_per_bbl": round(margin, 2),
            "product_value_per_bbl": round(product_value_per_bbl, 2),
            "margin_pct": round(margin_pct, 2),
            "crude_price": round(crude_price_bbl, 2),
            "crude_ratio": crude_ratio,
            "crude_label": crude_def["label"],
            "product_breakdown": product_contributions,
            "formula": config["formula"],
            "notes": config.get("notes"),
            "source": config.get("source"),
            "source_url": config.get("source_url"),
        }

    def analyze_regional_differentials(
        self,
        prices: dict[str, float]
    ) -> pd.DataFrame:
        """
        Analyze regional price differentials.

        Args:
            prices: Dictionary of regional prices

        Returns:
            DataFrame of differentials
        """
        # Default benchmark is Brent
        prices.get("Brent", prices.get("brent", 77.0))

        differentials = []

        # Common differentials
        diff_pairs = [
            ("WTI", "Brent", "WTI-Brent"),
            ("Dubai", "Brent", "Dubai-Brent"),
            ("WCS", "WTI", "WCS-WTI"),
            ("Mars", "WTI", "Mars-WTI"),
            ("LLS", "WTI", "LLS-WTI"),
        ]

        for grade1, grade2, name in diff_pairs:
            price1 = prices.get(grade1, prices.get(grade1.lower()))
            price2 = prices.get(grade2, prices.get(grade2.lower()))

            if price1 is not None and price2 is not None:
                diff = price1 - price2
                differentials.append({
                    "differential": name,
                    "grade_1": grade1,
                    "grade_2": grade2,
                    "price_1": round(price1, 2),
                    "price_2": round(price2, 2),
                    "spread": round(diff, 2),
                })

        return pd.DataFrame(differentials)

    def calculate_spread_zscore(
        self,
        current_spread: float,
        historical_spreads: pd.Series,
        lookback: int = 60
    ) -> dict:
        """
        Calculate z-score of spread vs historical.

        Args:
            current_spread: Current spread value
            historical_spreads: Historical spread values
            lookback: Lookback period in days

        Returns:
            Z-score analysis
        """
        if len(historical_spreads) < lookback:
            lookback = len(historical_spreads)

        recent = historical_spreads.tail(lookback)
        mean = recent.mean()
        std = recent.std()

        if std == 0:
            zscore = 0
        else:
            zscore = (current_spread - mean) / std

        # Signal interpretation
        if zscore > 2:
            signal = "Extremely Wide - Potential Short"
        elif zscore > 1:
            signal = "Wide - Watch for Mean Reversion"
        elif zscore < -2:
            signal = "Extremely Narrow - Potential Long"
        elif zscore < -1:
            signal = "Narrow - Watch for Mean Reversion"
        else:
            signal = "Normal Range"

        return {
            "current": round(current_spread, 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "zscore": round(zscore, 2),
            "percentile": round(stats.percentileofscore(recent, current_spread), 1),
            "signal": signal,
            "lookback_days": lookback,
        }

    def get_seasonal_pattern(
        self,
        spread_name: str,
        historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract seasonal pattern from spread history.

        Args:
            spread_name: Name of spread
            historical_data: Historical spread data with date index

        Returns:
            DataFrame with seasonal averages by month
        """
        if historical_data.empty:
            return pd.DataFrame()

        # Add month column
        df = historical_data.copy()
        df["month"] = df.index.month

        # Calculate monthly statistics
        seasonal = df.groupby("month").agg({
            "spread": ["mean", "std", "min", "max", "median"]
        }).round(2)

        seasonal.columns = ["mean", "std", "min", "max", "median"]
        seasonal["month_name"] = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        return seasonal

    def generate_spread_summary(
        self,
        wti: float,
        brent: float,
        rbob: float,
        ho: float
    ) -> dict:
        """
        Generate comprehensive spread summary.

        Args:
            wti: WTI price
            brent: Brent price
            rbob: RBOB price ($/gal)
            ho: Heating oil price ($/gal)

        Returns:
            Complete spread summary
        """
        wti_brent = self.calculate_wti_brent_spread(wti, brent)
        crack_321 = self.calculate_crack_spread(wti, rbob, ho, "3-2-1")
        gas_crack = self.calculate_crack_spread(wti, rbob, ho, "gasoline")
        ho_crack = self.calculate_crack_spread(wti, rbob, ho, "heating_oil")

        return {
            "wti_brent": wti_brent,
            "crack_321": crack_321,
            "gasoline_crack": gas_crack,
            "heating_oil_crack": ho_crack,
            "timestamp": datetime.now().isoformat(),
        }
