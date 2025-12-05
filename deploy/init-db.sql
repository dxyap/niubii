-- =============================================================================
-- Database Initialization Script
-- =============================================================================
-- Creates necessary schemas, tables, and initial data for the trading platform

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS analytics;

-- =============================================================================
-- Trading Schema
-- =============================================================================

-- Orders table
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 6) NOT NULL,
    price DECIMAL(18, 6),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    filled_quantity DECIMAL(18, 6) DEFAULT 0,
    avg_fill_price DECIMAL(18, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(50),
    strategy_id VARCHAR(50),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_created ON trading.orders(created_at);

-- Positions table
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(18, 6) NOT NULL DEFAULT 0,
    avg_entry_price DECIMAL(18, 6),
    current_price DECIMAL(18, 6),
    unrealized_pnl DECIMAL(18, 6),
    realized_pnl DECIMAL(18, 6) DEFAULT 0,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(50),
    strategy_id VARCHAR(50)
);

CREATE UNIQUE INDEX idx_positions_symbol_user ON trading.positions(symbol, user_id);

-- Trades table
CREATE TABLE IF NOT EXISTS trading.trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    order_id VARCHAR(50) REFERENCES trading.orders(order_id),
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18, 6) NOT NULL,
    price DECIMAL(18, 6) NOT NULL,
    commission DECIMAL(18, 6) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(50)
);

CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_executed ON trading.trades(executed_at);

-- =============================================================================
-- Audit Schema
-- =============================================================================

-- Audit events table
CREATE TABLE IF NOT EXISTS audit.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id VARCHAR(50),
    username VARCHAR(100),
    action TEXT NOT NULL,
    resource VARCHAR(100),
    resource_id VARCHAR(100),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

CREATE INDEX idx_audit_timestamp ON audit.events(timestamp);
CREATE INDEX idx_audit_event_type ON audit.events(event_type);
CREATE INDEX idx_audit_user_id ON audit.events(user_id);

-- =============================================================================
-- Analytics Schema
-- =============================================================================

-- Price history table
CREATE TABLE IF NOT EXISTS analytics.price_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(18, 6),
    high DECIMAL(18, 6),
    low DECIMAL(18, 6),
    close DECIMAL(18, 6),
    volume DECIMAL(18, 6),
    source VARCHAR(50) DEFAULT 'bloomberg'
);

CREATE INDEX idx_price_symbol_time ON analytics.price_history(symbol, timestamp);

-- Signals table
CREATE TABLE IF NOT EXISTS analytics.signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(50) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    signal_value DECIMAL(10, 4),
    direction VARCHAR(10),
    confidence DECIMAL(5, 2),
    source VARCHAR(50),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_signals_symbol_time ON analytics.signals(symbol, timestamp);

-- ML predictions table
CREATE TABLE IF NOT EXISTS analytics.ml_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    symbol VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prediction DECIMAL(10, 4),
    probability DECIMAL(5, 4),
    horizon_days INTEGER,
    features JSONB,
    actual_outcome DECIMAL(10, 4)
);

CREATE INDEX idx_predictions_symbol_time ON analytics.ml_predictions(symbol, timestamp);

-- =============================================================================
-- Users and Authentication
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    roles TEXT[] DEFAULT ARRAY['viewer'],
    is_active BOOLEAN DEFAULT true,
    is_locked BOOLEAN DEFAULT false,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trading.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES trading.users(id),
    token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_sessions_token ON trading.sessions(token);
CREATE INDEX idx_sessions_user_id ON trading.sessions(user_id);

-- =============================================================================
-- Alerts and Notifications
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading.alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_alerts_status ON trading.alerts(status);
CREATE INDEX idx_alerts_timestamp ON trading.alerts(timestamp);

-- =============================================================================
-- Risk Management
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading.risk_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    limit_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(50),
    value DECIMAL(18, 6) NOT NULL,
    current_value DECIMAL(18, 6) DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS trading.risk_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    limit_id UUID REFERENCES trading.risk_limits(id),
    limit_value DECIMAL(18, 6),
    current_value DECIMAL(18, 6),
    breach_pct DECIMAL(10, 4),
    action_taken VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_risk_events_timestamp ON trading.risk_events(timestamp);

-- =============================================================================
-- Functions and Triggers
-- =============================================================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables with updated_at
CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON trading.orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON trading.positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON trading.users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert default admin user (password: admin123!)
INSERT INTO trading.users (username, email, password_hash, roles)
VALUES (
    'admin',
    'admin@localhost',
    'default_hash_change_me',
    ARRAY['admin']
) ON CONFLICT (username) DO NOTHING;

-- Insert default risk limits
INSERT INTO trading.risk_limits (limit_type, symbol, value)
VALUES
    ('max_position_size', NULL, 1000000),
    ('max_daily_loss', NULL, 50000),
    ('max_var_95', NULL, 100000),
    ('max_leverage', NULL, 2.0)
ON CONFLICT DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA trading TO PUBLIC;
GRANT USAGE ON SCHEMA audit TO PUBLIC;
GRANT USAGE ON SCHEMA analytics TO PUBLIC;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO PUBLIC;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO PUBLIC;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO PUBLIC;
