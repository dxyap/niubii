"""Initial schema

Revision ID: 20241205_000001
Revises: 
Create Date: 2024-12-05

Initial database schema for the Oil Trading Dashboard.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision: str = '20241205_000001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create schemas
    op.execute('CREATE SCHEMA IF NOT EXISTS trading')
    op.execute('CREATE SCHEMA IF NOT EXISTS audit')
    op.execute('CREATE SCHEMA IF NOT EXISTS analytics')
    
    # =========================================================================
    # Trading Schema Tables
    # =========================================================================
    
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('username', sa.String(100), unique=True, nullable=False),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('roles', postgresql.ARRAY(sa.String), server_default='{}'),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('is_locked', sa.Boolean, server_default='false'),
        sa.Column('failed_attempts', sa.Integer, server_default='0'),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        schema='trading'
    )
    
    # Sessions table
    op.create_table(
        'sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('trading.users.id'), nullable=False),
        sa.Column('token', sa.String(255), unique=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ip_address', postgresql.INET, nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        schema='trading'
    )
    op.create_index('idx_sessions_token', 'sessions', ['token'], schema='trading')
    op.create_index('idx_sessions_user_id', 'sessions', ['user_id'], schema='trading')
    
    # Orders table
    op.create_table(
        'orders',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('order_id', sa.String(50), unique=True, nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('order_type', sa.String(20), nullable=False),
        sa.Column('quantity', sa.Numeric(18, 6), nullable=False),
        sa.Column('price', sa.Numeric(18, 6), nullable=True),
        sa.Column('status', sa.String(20), server_default='pending'),
        sa.Column('filled_quantity', sa.Numeric(18, 6), server_default='0'),
        sa.Column('avg_fill_price', sa.Numeric(18, 6), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('user_id', sa.String(50), nullable=True),
        sa.Column('strategy_id', sa.String(50), nullable=True),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        sa.CheckConstraint("side IN ('buy', 'sell')"),
        schema='trading'
    )
    op.create_index('idx_orders_symbol', 'orders', ['symbol'], schema='trading')
    op.create_index('idx_orders_status', 'orders', ['status'], schema='trading')
    op.create_index('idx_orders_created', 'orders', ['created_at'], schema='trading')
    
    # Positions table
    op.create_table(
        'positions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('quantity', sa.Numeric(18, 6), server_default='0'),
        sa.Column('avg_entry_price', sa.Numeric(18, 6), nullable=True),
        sa.Column('current_price', sa.Numeric(18, 6), nullable=True),
        sa.Column('unrealized_pnl', sa.Numeric(18, 6), nullable=True),
        sa.Column('realized_pnl', sa.Numeric(18, 6), server_default='0'),
        sa.Column('opened_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('user_id', sa.String(50), nullable=True),
        sa.Column('strategy_id', sa.String(50), nullable=True),
        schema='trading'
    )
    op.create_index('idx_positions_symbol_user', 'positions', ['symbol', 'user_id'], unique=True, schema='trading')
    
    # Trades table
    op.create_table(
        'trades',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('trade_id', sa.String(50), unique=True, nullable=False),
        sa.Column('order_id', sa.String(50), nullable=True),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('side', sa.String(10), nullable=False),
        sa.Column('quantity', sa.Numeric(18, 6), nullable=False),
        sa.Column('price', sa.Numeric(18, 6), nullable=False),
        sa.Column('commission', sa.Numeric(18, 6), server_default='0'),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('user_id', sa.String(50), nullable=True),
        schema='trading'
    )
    op.create_index('idx_trades_symbol', 'trades', ['symbol'], schema='trading')
    op.create_index('idx_trades_executed', 'trades', ['executed_at'], schema='trading')
    
    # Alerts table
    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('rule_id', sa.String(100), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), server_default='active'),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('acknowledged_by', sa.String(100), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.String(100), nullable=True),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        schema='trading'
    )
    op.create_index('idx_alerts_status', 'alerts', ['status'], schema='trading')
    op.create_index('idx_alerts_timestamp', 'alerts', ['timestamp'], schema='trading')
    
    # Risk limits table
    op.create_table(
        'risk_limits',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('limit_type', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=True),
        sa.Column('value', sa.Numeric(18, 6), nullable=False),
        sa.Column('current_value', sa.Numeric(18, 6), server_default='0'),
        sa.Column('is_active', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('created_by', sa.String(100), nullable=True),
        schema='trading'
    )
    
    # =========================================================================
    # Audit Schema Tables
    # =========================================================================
    
    op.create_table(
        'events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('user_id', sa.String(50), nullable=True),
        sa.Column('username', sa.String(100), nullable=True),
        sa.Column('action', sa.Text, nullable=False),
        sa.Column('resource', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('details', postgresql.JSONB, server_default='{}'),
        sa.Column('ip_address', postgresql.INET, nullable=True),
        sa.Column('user_agent', sa.Text, nullable=True),
        sa.Column('success', sa.Boolean, server_default='true'),
        sa.Column('error_message', sa.Text, nullable=True),
        schema='audit'
    )
    op.create_index('idx_audit_timestamp', 'events', ['timestamp'], schema='audit')
    op.create_index('idx_audit_event_type', 'events', ['event_type'], schema='audit')
    op.create_index('idx_audit_user_id', 'events', ['user_id'], schema='audit')
    
    # =========================================================================
    # Analytics Schema Tables
    # =========================================================================
    
    # Price history table
    op.create_table(
        'price_history',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('open', sa.Numeric(18, 6), nullable=True),
        sa.Column('high', sa.Numeric(18, 6), nullable=True),
        sa.Column('low', sa.Numeric(18, 6), nullable=True),
        sa.Column('close', sa.Numeric(18, 6), nullable=True),
        sa.Column('volume', sa.Numeric(18, 6), nullable=True),
        sa.Column('source', sa.String(50), server_default='bloomberg'),
        schema='analytics'
    )
    op.create_index('idx_price_symbol_time', 'price_history', ['symbol', 'timestamp'], schema='analytics')
    
    # Signals table
    op.create_table(
        'signals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('signal_type', sa.String(50), nullable=False),
        sa.Column('signal_value', sa.Numeric(10, 4), nullable=True),
        sa.Column('direction', sa.String(10), nullable=True),
        sa.Column('confidence', sa.Numeric(5, 2), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('metadata', postgresql.JSONB, server_default='{}'),
        schema='analytics'
    )
    op.create_index('idx_signals_symbol_time', 'signals', ['symbol', 'timestamp'], schema='analytics')
    
    # ML predictions table
    op.create_table(
        'ml_predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('prediction', sa.Numeric(10, 4), nullable=True),
        sa.Column('probability', sa.Numeric(5, 4), nullable=True),
        sa.Column('horizon_days', sa.Integer, nullable=True),
        sa.Column('features', postgresql.JSONB, nullable=True),
        sa.Column('actual_outcome', sa.Numeric(10, 4), nullable=True),
        schema='analytics'
    )
    op.create_index('idx_predictions_symbol_time', 'ml_predictions', ['symbol', 'timestamp'], schema='analytics')
    
    # =========================================================================
    # Triggers
    # =========================================================================
    
    # Create update timestamp function
    op.execute('''
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    ''')
    
    # Apply triggers
    for table in ['orders', 'positions', 'users', 'risk_limits']:
        op.execute(f'''
            CREATE TRIGGER update_{table}_updated_at
                BEFORE UPDATE ON trading.{table}
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        ''')


def downgrade() -> None:
    """Drop all tables and schemas."""
    
    # Drop triggers
    for table in ['orders', 'positions', 'users', 'risk_limits']:
        op.execute(f'DROP TRIGGER IF EXISTS update_{table}_updated_at ON trading.{table}')
    
    op.execute('DROP FUNCTION IF EXISTS update_updated_at_column()')
    
    # Drop analytics tables
    op.drop_index('idx_predictions_symbol_time', table_name='ml_predictions', schema='analytics')
    op.drop_table('ml_predictions', schema='analytics')
    
    op.drop_index('idx_signals_symbol_time', table_name='signals', schema='analytics')
    op.drop_table('signals', schema='analytics')
    
    op.drop_index('idx_price_symbol_time', table_name='price_history', schema='analytics')
    op.drop_table('price_history', schema='analytics')
    
    # Drop audit tables
    op.drop_index('idx_audit_user_id', table_name='events', schema='audit')
    op.drop_index('idx_audit_event_type', table_name='events', schema='audit')
    op.drop_index('idx_audit_timestamp', table_name='events', schema='audit')
    op.drop_table('events', schema='audit')
    
    # Drop trading tables
    op.drop_table('risk_limits', schema='trading')
    
    op.drop_index('idx_alerts_timestamp', table_name='alerts', schema='trading')
    op.drop_index('idx_alerts_status', table_name='alerts', schema='trading')
    op.drop_table('alerts', schema='trading')
    
    op.drop_index('idx_trades_executed', table_name='trades', schema='trading')
    op.drop_index('idx_trades_symbol', table_name='trades', schema='trading')
    op.drop_table('trades', schema='trading')
    
    op.drop_index('idx_positions_symbol_user', table_name='positions', schema='trading')
    op.drop_table('positions', schema='trading')
    
    op.drop_index('idx_orders_created', table_name='orders', schema='trading')
    op.drop_index('idx_orders_status', table_name='orders', schema='trading')
    op.drop_index('idx_orders_symbol', table_name='orders', schema='trading')
    op.drop_table('orders', schema='trading')
    
    op.drop_index('idx_sessions_user_id', table_name='sessions', schema='trading')
    op.drop_index('idx_sessions_token', table_name='sessions', schema='trading')
    op.drop_table('sessions', schema='trading')
    
    op.drop_table('users', schema='trading')
    
    # Drop schemas
    op.execute('DROP SCHEMA IF EXISTS analytics CASCADE')
    op.execute('DROP SCHEMA IF EXISTS audit CASCADE')
    op.execute('DROP SCHEMA IF EXISTS trading CASCADE')
