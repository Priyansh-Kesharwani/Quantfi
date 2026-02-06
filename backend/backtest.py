import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from models import BacktestConfig, BacktestResult
from data_providers import PriceProvider
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """DCA Backtesting Engine"""
    
    @classmethod
    def run_backtest(cls, config: BacktestConfig, historical_data: pd.DataFrame, scores: pd.DataFrame = None) -> BacktestResult:
        """
        Run DCA backtest
        historical_data: DataFrame with Date, Close, and optionally Score columns
        scores: DataFrame with Date and Score columns (optional)
        """
        # Merge scores if provided
        if scores is not None and not scores.empty:
            historical_data = historical_data.join(scores[['Score']], how='left')
            historical_data['Score'] = historical_data['Score'].fillna(50)  # Neutral default
        else:
            historical_data['Score'] = 50
        
        # Filter date range
        mask = (historical_data.index >= config.start_date) & (historical_data.index <= config.end_date)
        data = historical_data[mask].copy()
        
        if data.empty:
            raise ValueError("No data in specified date range")
        
        # Determine DCA dates
        dca_dates = cls._get_dca_dates(config.start_date, config.end_date, config.dca_cadence)
        
        total_invested = 0
        total_units = 0
        num_regular_dca = 0
        num_dip_buys = 0
        
        for dca_date in dca_dates:
            # Find nearest trading day
            available_dates = data.index[data.index >= dca_date]
            if len(available_dates) == 0:
                continue
            
            trade_date = available_dates[0]
            price = data.loc[trade_date, 'Close']
            score = data.loc[trade_date, 'Score']
            
            # Regular DCA
            units_bought = config.dca_amount / price
            total_units += units_bought
            total_invested += config.dca_amount
            num_regular_dca += 1
            
            # Extra dip buying if score is favorable
            if config.buy_dip_threshold and score >= config.buy_dip_threshold:
                dip_amount = config.dca_amount * 0.5  # Extra 50% on dips
                dip_units = dip_amount / price
                total_units += dip_units
                total_invested += dip_amount
                num_dip_buys += 1
        
        # Calculate final value
        final_price = data.iloc[-1]['Close']
        final_value_usd = total_units * final_price
        
        # Get USD-INR rate (simplified - using current rate)
        # In production, would use historical rates
        from data_providers import FXProvider
        usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5
        final_value_inr = final_value_usd * usd_inr_rate
        
        # Calculate returns
        total_return_pct = ((final_value_usd - total_invested) / total_invested) * 100 if total_invested > 0 else 0
        
        # Annualized return
        days = (config.end_date - config.start_date).days
        years = days / 365.25
        annualized_return_pct = (((final_value_usd / total_invested) ** (1 / years)) - 1) * 100 if years > 0 and total_invested > 0 else 0
        
        return BacktestResult(
            symbol=config.symbol,
            config=config,
            total_invested=total_invested,
            total_units=total_units,
            final_value_usd=final_value_usd,
            final_value_inr=final_value_inr,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            num_regular_dca=num_regular_dca,
            num_dip_buys=num_dip_buys
        )
    
    @staticmethod
    def _get_dca_dates(start_date: datetime, end_date: datetime, cadence: str) -> List[datetime]:
        """Generate list of DCA dates based on cadence"""
        dates = []
        current = start_date
        
        if cadence == 'weekly':
            while current <= end_date:
                dates.append(current)
                current += timedelta(days=7)
        elif cadence == 'monthly':
            while current <= end_date:
                dates.append(current)
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        
        return dates
