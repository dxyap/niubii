"""
Trade Blotter
=============
Trade entry, storage, and history management.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid
import json


class TradeBlotter:
    """
    Trade blotter for recording and managing trades.
    
    Features:
    - Manual trade entry
    - Trade history storage (SQLite)
    - Trade querying and filtering
    - Trade statistics
    """
    
    def __init__(self, db_path: str = "data/trades/trades.db"):
        """
        Initialize trade blotter.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                trade_date DATE NOT NULL,
                trade_time TIME NOT NULL,
                instrument TEXT NOT NULL,
                side TEXT CHECK(side IN ('BUY', 'SELL')),
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                notional REAL,
                commission REAL DEFAULT 0,
                strategy TEXT,
                signal_ref TEXT,
                notes TEXT,
                account TEXT DEFAULT 'MAIN',
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily P&L table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date DATE PRIMARY KEY,
                starting_nav REAL,
                ending_nav REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                total_pnl REAL,
                commissions REAL,
                num_trades INTEGER
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
        
        conn.commit()
        conn.close()
    
    def add_trade(
        self,
        instrument: str,
        side: str,
        quantity: int,
        price: float,
        trade_date: date = None,
        trade_time: str = None,
        commission: float = 0,
        strategy: str = None,
        signal_ref: str = None,
        notes: str = None,
        account: str = "MAIN"
    ) -> str:
        """
        Add a new trade to the blotter.
        
        Args:
            instrument: Trading instrument
            side: 'BUY' or 'SELL'
            quantity: Number of contracts
            price: Execution price
            trade_date: Trade date (default: today)
            trade_time: Trade time (default: now)
            commission: Commission/fees
            strategy: Strategy tag
            signal_ref: Signal reference ID
            notes: Trade notes
            account: Trading account
            
        Returns:
            Trade ID
        """
        if trade_date is None:
            trade_date = datetime.now().date()
        if trade_time is None:
            trade_time = datetime.now().strftime("%H:%M:%S")
        
        trade_id = f"TRD-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
        
        # Calculate notional
        contract_type = instrument[:2]
        multiplier = 1000 if contract_type in ["CL", "CO"] else 42000
        notional = quantity * price * multiplier
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                trade_id, trade_date, trade_time, instrument, side,
                quantity, price, notional, commission, strategy,
                signal_ref, notes, account
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, trade_date.isoformat(), trade_time, instrument, side.upper(),
            quantity, price, notional, commission, strategy,
            signal_ref, notes, account
        ))
        
        conn.commit()
        conn.close()
        
        return trade_id
    
    def get_trades(
        self,
        start_date: date = None,
        end_date: date = None,
        instrument: str = None,
        strategy: str = None,
        status: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Query trades with filters.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            instrument: Filter by instrument
            strategy: Filter by strategy
            status: Filter by status
            limit: Maximum number of trades
            
        Returns:
            DataFrame of trades
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date.isoformat())
        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY trade_date DESC, trade_time DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Dict]:
        """Get single trade by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            trade = dict(zip(columns, row))
        else:
            trade = None
        
        conn.close()
        return trade
    
    def update_trade(self, trade_id: str, updates: Dict) -> bool:
        """
        Update an existing trade.
        
        Args:
            trade_id: Trade ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful
        """
        allowed_fields = ["price", "quantity", "commission", "strategy", "notes", "status"]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for field, value in updates.items():
            if field in allowed_fields:
                cursor.execute(
                    f"UPDATE trades SET {field} = ? WHERE trade_id = ?",
                    (value, trade_id)
                )
        
        conn.commit()
        conn.close()
        return True
    
    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM trades WHERE trade_id = ?", (trade_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def get_trade_statistics(
        self,
        start_date: date = None,
        end_date: date = None,
        strategy: str = None
    ) -> Dict:
        """
        Calculate trade statistics.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            strategy: Strategy filter
            
        Returns:
            Trade statistics dictionary
        """
        trades = self.get_trades(
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            limit=10000
        )
        
        if trades.empty:
            return {
                "total_trades": 0,
                "total_volume": 0,
                "total_commission": 0,
            }
        
        # Basic stats
        total_trades = len(trades)
        buy_trades = len(trades[trades["side"] == "BUY"])
        sell_trades = len(trades[trades["side"] == "SELL"])
        
        total_volume = trades["quantity"].sum()
        total_commission = trades["commission"].sum()
        total_notional = trades["notional"].abs().sum()
        
        # By strategy
        by_strategy = trades.groupby("strategy").agg({
            "trade_id": "count",
            "quantity": "sum",
            "commission": "sum",
        }).to_dict("index")
        
        # By instrument
        by_instrument = trades.groupby("instrument").agg({
            "trade_id": "count",
            "quantity": "sum",
        }).to_dict("index")
        
        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_volume": int(total_volume),
            "total_commission": round(total_commission, 2),
            "total_notional": round(total_notional, 2),
            "avg_trade_size": round(total_volume / total_trades, 1) if total_trades > 0 else 0,
            "by_strategy": by_strategy,
            "by_instrument": by_instrument,
        }
    
    def export_trades(
        self,
        filepath: str,
        format: str = "csv",
        start_date: date = None,
        end_date: date = None
    ) -> bool:
        """
        Export trades to file.
        
        Args:
            filepath: Output file path
            format: 'csv' or 'json'
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            True if successful
        """
        trades = self.get_trades(
            start_date=start_date,
            end_date=end_date,
            limit=100000
        )
        
        if format == "csv":
            trades.to_csv(filepath, index=False)
        elif format == "json":
            trades.to_json(filepath, orient="records", date_format="iso")
        else:
            return False
        
        return True
    
    def get_todays_trades(self) -> pd.DataFrame:
        """Get all trades from today."""
        return self.get_trades(
            start_date=datetime.now().date(),
            end_date=datetime.now().date()
        )
