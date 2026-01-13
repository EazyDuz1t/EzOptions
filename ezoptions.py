import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
from math import log, sqrt
import re
import time
from scipy.stats import norm
import threading
import concurrent.futures
from contextlib import contextmanager
from scipy.interpolate import griddata
import numpy as np
import pytz
from datetime import timedelta
import requests
import json
from io import StringIO
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except ImportError:
    try:
        from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx  # type: ignore
    except ImportError:
        add_script_run_ctx = None
        get_script_run_ctx = None


def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candlestick values."""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA_Open with first candle's opening price
    ha_df['HA_Open'] = pd.Series(index=df.index)
    ha_df.loc[ha_df.index[0], 'HA_Open'] = df['Open'].iloc[0]
    
    # Calculate subsequent HA_Open values
    for i in range(1, len(df)):
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df


def calculate_technical_indicators(df):
    """Calculate various technical indicators for intraday data."""
    if df is None or len(df) == 0:
        return {}
    
    indicators = {}
    selected_indicators = st.session_state.get('selected_indicators', [])
    
    # EMA calculations
    if "EMA (Exponential Moving Average)" in selected_indicators and st.session_state.get('ema_periods'):
        indicators['ema'] = {}
        for period in st.session_state.ema_periods:
            if len(df) >= period:
                indicators['ema'][period] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # SMA calculations
    if "SMA (Simple Moving Average)" in selected_indicators and st.session_state.get('sma_periods'):
        indicators['sma'] = {}
        for period in st.session_state.sma_periods:
            if len(df) >= period:
                indicators['sma'][period] = df['Close'].rolling(window=period).mean()
    
    # Bollinger Bands
    if "Bollinger Bands" in selected_indicators and st.session_state.get('bollinger_period'):
        period = st.session_state.bollinger_period
        std_dev = st.session_state.get('bollinger_std', 2.0)
        if len(df) >= period:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            indicators['bollinger'] = {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
    
    # RSI
    if "RSI (Relative Strength Index)" in selected_indicators and st.session_state.get('rsi_period'):
        period = st.session_state.rsi_period
        if len(df) >= period + 1:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # VWAP
    if "VWAP (Volume Weighted Average Price)" in selected_indicators and 'Volume' in df.columns:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_price_volume = (typical_price * df['Volume']).cumsum()
        indicators['vwap'] = cumulative_price_volume / cumulative_volume
    
    return indicators


def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels based on recent high and low."""
    if df is None or len(df) == 0:
        return {}
    
    # Use the full range for fibonacci calculation
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    
    fibonacci_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    
    for ratio in fibonacci_ratios:
        levels[f"{ratio:.3f}"] = high - (diff * ratio)
    
    return levels


def add_technical_indicators_to_chart(fig, indicators, fibonacci_levels=None):
    """Add technical indicators to the intraday chart."""
    if not indicators:
        return fig
    
    # Color palette for indicators
    colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0E68C']
    color_index = 0
    
    # Add EMAs
    if 'ema' in indicators:
        ema_periods = list(indicators['ema'].keys())
        for i, (period, ema_data) in enumerate(indicators['ema'].items()):
            fig.add_trace(
                go.Scatter(
                    x=ema_data.index,
                    y=ema_data.values,
                    mode='lines',
                    name=f'EMA {period}',
                    line=dict(color=colors[color_index % len(colors)], width=2),
                    opacity=0.8
                ),
                secondary_y=False
            )
            color_index += 1
        
        # Add EMA cloud if we have at least 2 EMAs
        if len(ema_periods) >= 2:
            # Create cloud between first two EMAs (typically fastest and slower)
            ema1 = indicators['ema'][ema_periods[0]]
            ema2 = indicators['ema'][ema_periods[1]]
            
            # Determine which EMA is above/below for coloring
            fig.add_trace(
                go.Scatter(
                    x=ema1.index,
                    y=ema1.values,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                    showlegend=False,
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=ema2.index,
                    y=ema2.values,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                    fill='tonexty',
                    fillcolor='rgba(100, 150, 200, 0.1)',  # Light blue cloud
                    name=f'EMA Cloud ({ema_periods[0]}-{ema_periods[1]})',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
    
    # Add SMAs
    if 'sma' in indicators:
        for period, sma_data in indicators['sma'].items():
            fig.add_trace(
                go.Scatter(
                    x=sma_data.index,
                    y=sma_data.values,
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(color=colors[color_index % len(colors)], width=2, dash='dash'),
                    opacity=0.8
                ),
                secondary_y=False
            )
            color_index += 1
    
    # Add Bollinger Bands
    if 'bollinger' in indicators:
        bb = indicators['bollinger']
        # Add middle line first
        fig.add_trace(
            go.Scatter(
                x=bb['middle'].index,
                y=bb['middle'].values,
                mode='lines',
                name=f'BB Middle ({st.session_state.bollinger_period})',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.7
            ),
            secondary_y=False
        )
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=bb['upper'].index,
                y=bb['upper'].values,
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                showlegend=False
            ),
            secondary_y=False
        )
        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=bb['lower'].index,
                y=bb['lower'].values,
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False
            ),
            secondary_y=False
        )
    
    # Add VWAP
    if 'vwap' in indicators:
        fig.add_trace(
            go.Scatter(
                x=indicators['vwap'].index,
                y=indicators['vwap'].values,
                mode='lines',
                name='VWAP',
                line=dict(color='orange', width=2, dash='dashdot'),
                opacity=0.8
            ),
            secondary_y=False
        )
    
    # Add Fibonacci levels as horizontal lines
    if fibonacci_levels:
        for level_name, level_value in fibonacci_levels.items():
            fig.add_hline(
                y=level_value,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.4)",
                annotation_text=f"Fib {level_name}",
                annotation_position="right",
                annotation_font_size=st.session_state.chart_text_size
            )
    
    return fig


@contextmanager
def st_thread_context():
    """Thread context management for Streamlit"""
    try:
        if not hasattr(threading.current_thread(), '_StreamlitThread__cached_st'):
           
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*missing ScriptRunContext.*')
        yield
    finally:
        pass


with st_thread_context():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Prevent page dimming during reruns
st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)

# Force a rerun on first load to ensure clean state
if 'loading_complete' not in st.session_state:
    st.session_state.loading_complete = False

if not st.session_state.loading_complete:
    st.session_state.loading_complete = True
    st.rerun()

# Initialize session state for colors if not already set
if 'call_color' not in st.session_state:
    st.session_state.call_color = '#00FF00'  # Default green for calls
if 'put_color' not in st.session_state:
    st.session_state.put_color = '#FF0000'   # Default red for puts
if 'vix_color' not in st.session_state:
    st.session_state.vix_color = '#800080'   # Default purple for VIXY

# -------------------------------
# Helper Functions
# -------------------------------
def format_ticker(ticker):
    """Helper function to format tickers for indices"""
    ticker = ticker.upper()
    if ticker == "SPX":
        return "^SPX"
    elif ticker == "NDX":
        return "^NDX"
    elif ticker == "VIX":
        return "VIXY"
    elif ticker == "DJI":
        return "^DJI"
    elif ticker == "RUT":
        return "^RUT"
    return ticker

def format_large_number(num):
    """Format large numbers with suffixes (K, M, B, T)"""
    if num is None:
        return "0"
    
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:,.0f}"

def get_now_et():
    """Get current time in US/Eastern timezone"""
    et_tz = pytz.timezone('US/Eastern')
    return datetime.now(et_tz)

def calculate_time_to_expiration(expiry_date):
    """
    Calculate time to expiration in years using Eastern Time.
    expiry_date: datetime.date object or string 'YYYY-MM-DD'
    Returns: time in years (float)
    """
    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        if isinstance(expiry_date, str):
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        elif isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()
            
        # Set expiration to 4:00 PM ET on the expiration date
        expiry_dt = datetime.combine(expiry_date, datetime.min.time()) + timedelta(hours=16)
        expiry_dt = et_tz.localize(expiry_dt)
        
        # Calculate time difference in years
        diff = expiry_dt - now_et
        t = diff.total_seconds() / (365 * 24 * 3600)
        
        return t
             
    except Exception as e:
        print(f"Error calculating time to expiration: {e}")
        return 0

def check_market_status():
    """Check if we're in pre-market, market hours, or post-market"""
    # Get current time in ET for market checks
    eastern = get_now_et()
    
    # Get local time and timezone
    local = datetime.now()
    local_tz = datetime.now().astimezone().tzinfo
    
    market_message = None
    
    # Check for "Wait for new data" window (12 AM ET to 10 AM ET)
    # This corresponds to 9 PM PT to 7 AM PT in the original code
    if eastern.hour < 10:
        next_update = eastern.replace(hour=10, minute=0)
        time_until = next_update - eastern
        hours, remainder = divmod(time_until.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        # Convert ET update time to local time
        local_next_update = next_update.astimezone(local_tz)
        
        market_message = f"""
        ⚠️ **WAIT FOR NEW DATA**
        - Current time: {local.strftime('%I:%M %p')} {local_tz}
        - New data will be available at approximately {local_next_update.strftime('%I:%M %p')}
        - Time until new data: {hours}h {minutes}m
        """
    return market_message

def get_cache_ttl():
    """Get the cache TTL from session state refresh rate, with a minimum of 10 seconds"""
    return max(float(st.session_state.get('refresh_rate', 10)), 10)

def calculate_strike_range(current_price, percentage=None):
    """Calculate strike range based on percentage of current price"""
    if percentage is None:
        percentage = st.session_state.get('strike_range', 1.0)
    return current_price * (percentage / 100.0)

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)  # Cache TTL matches refresh rate
def fetch_options_for_date(ticker, date, S=None):
    """Fetch options data for a specific date with caching"""
    if ticker == "MARKET":
        # Get prices for scaling (Base price is ^GSPC)
        spx_price = S if S else get_current_price("^SPX")
        spy_price = get_current_price("SPY")
        qqq_price = get_current_price("QQQ")
        iwm_price = get_current_price("IWM")
        
        if not (spx_price and spy_price and qqq_price and iwm_price):
             # Try our best with what we have, but SPX is required for scaling base
             if not spx_price:
                 return pd.DataFrame(), pd.DataFrame()
        
        # Calculate scaling factors (Target Base / Source Base)
        # We scale strikes of SPY/QQQ/IWM up to SPX levels
        spy_factor = spx_price / spy_price if spy_price else 0
        qqq_factor = spx_price / qqq_price if qqq_price else 0
        iwm_factor = spx_price / iwm_price if iwm_price else 0
        
        calls_list = []
        puts_list = []
        spx_strikes_grid = None
        
        # Helper to fetch and scale
        def add_scaled_data(tick, price, factor):
            try:
                # Recursive call to fetch data for component ticker
                c, p = fetch_options_for_date(tick, date, price)
                
                cols_to_scale_up = ['lastPrice', 'bid', 'ask', 'change', 'strike']
                cols_to_scale_down = ['volume', 'openInterest']
                
                if not c.empty:
                    c = c.copy()
                    if factor != 1.0:
                        for col in cols_to_scale_up:
                            if col in c.columns:
                                c[col] = c[col] * factor
                        for col in cols_to_scale_down:
                            if col in c.columns:
                                c[col] = c[col] / factor
                        
                        # Snap to SPX grid if available
                        if spx_strikes_grid is not None and len(spx_strikes_grid) > 0 and 'strike' in c.columns:
                            vals = c['strike'].values
                            # Find nearest strike in grid
                            idx = np.abs(spx_strikes_grid[None, :] - vals[:, None]).argmin(axis=1)
                            c['strike'] = spx_strikes_grid[idx]
                        elif 'strike' in c.columns:
                             # Fallback to rounding if grid is somehow missing for non-1.0 factor
                             c['strike'] = c['strike'].round(0).astype(int)
                    
                    calls_list.append(c)
                
                if not p.empty:
                    p = p.copy()
                    if factor != 1.0:
                        for col in cols_to_scale_up:
                            if col in p.columns:
                                p[col] = p[col] * factor
                        for col in cols_to_scale_down:
                             if col in p.columns:
                                p[col] = p[col] / factor
                        
                        # Snap to SPX grid if available
                        if spx_strikes_grid is not None and len(spx_strikes_grid) > 0 and 'strike' in p.columns:
                            vals = p['strike'].values
                            # Find nearest strike in grid
                            idx = np.abs(spx_strikes_grid[None, :] - vals[:, None]).argmin(axis=1)
                            p['strike'] = spx_strikes_grid[idx]
                        elif 'strike' in p.columns:
                            p['strike'] = p['strike'].round(0).astype(int)

                    puts_list.append(p)
            except Exception:
                pass # Date might not exist for this ticker

        # Build reference grid from SPX data and add SPX options to the result
        grid_strikes = set()
        try:
            # Fetch SPX data to get the strikes grid
            spx_c, spx_p = fetch_options_for_date("^SPX", date, spx_price)
            if not spx_c.empty and 'strike' in spx_c.columns:
                grid_strikes.update(spx_c['strike'].tolist())
                # Add SPX calls to result (no scaling needed, factor = 1.0)
                calls_list.append(spx_c)
            if not spx_p.empty and 'strike' in spx_p.columns:
                grid_strikes.update(spx_p['strike'].tolist())
                # Add SPX puts to result (no scaling needed, factor = 1.0)
                puts_list.append(spx_p)
        except Exception:
            pass
            
        if grid_strikes:
            spx_strikes_grid = np.array(sorted(list(grid_strikes)))
        
        if spy_factor: add_scaled_data("SPY", spy_price, spy_factor)
        if qqq_factor: add_scaled_data("QQQ", qqq_price, qqq_factor)
        if iwm_factor: add_scaled_data("IWM", iwm_price, iwm_factor)
        
        combined_calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
        combined_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()
        
        return combined_calls, combined_puts

    print(f"Fetching option chain for {ticker} EXP {date}")
    try:
        stock = get_ticker_object(ticker)
        chain = stock.option_chain(date)
        calls = chain.calls
        puts = chain.puts
        
        if not calls.empty:
            calls = calls.copy()
            calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        if not puts.empty:
            puts = puts.copy()
            puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
            
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)  # Cache TTL matches refresh rate
def fetch_all_options(ticker):
    """Fetch all available options with caching"""
    print(f"Fetching all options for {ticker}")
    
    if ticker == "MARKET":
        spx_price = get_current_price("^SPX")
        if not spx_price: return pd.DataFrame(), pd.DataFrame()
        
        spy_price = get_current_price("SPY")
        qqq_price = get_current_price("QQQ") 
        iwm_price = get_current_price("IWM")
        
        spy_factor = spx_price / spy_price if spy_price else 1
        qqq_factor = spx_price / qqq_price if qqq_price else 1
        iwm_factor = spx_price / iwm_price if iwm_price else 1
        
        calls_list = []
        puts_list = []
        spx_strikes_grid = None
        
        # Helper to fetch and scale
        def process_component(tick, factor):
            try:
                c, p = fetch_all_options(tick)
                cols_to_scale_up = ['lastPrice', 'bid', 'ask', 'change', 'strike']
                cols_to_scale_down = ['volume', 'openInterest']
                
                if not c.empty:
                    c = c.copy()
                    if factor != 1.0: 
                        for col in cols_to_scale_up:
                            if col in c.columns:
                                c[col] = c[col] * factor
                        for col in cols_to_scale_down:
                             if col in c.columns:
                                c[col] = c[col] / factor
                    
                        # Snap to SPX grid if available
                        if spx_strikes_grid is not None and len(spx_strikes_grid) > 0 and 'strike' in c.columns:
                            vals = c['strike'].values
                            idx = np.abs(spx_strikes_grid[None, :] - vals[:, None]).argmin(axis=1)
                            c['strike'] = spx_strikes_grid[idx]
                        elif 'strike' in c.columns:
                            c['strike'] = c['strike'].round(0).astype(int)
                    
                    calls_list.append(c)
                if not p.empty:
                    p = p.copy()
                    if factor != 1.0: 
                        for col in cols_to_scale_up:
                            if col in p.columns:
                                p[col] = p[col] * factor
                        for col in cols_to_scale_down:
                             if col in p.columns:
                                p[col] = p[col] / factor
                    
                        # Snap to SPX grid if available
                        if spx_strikes_grid is not None and len(spx_strikes_grid) > 0 and 'strike' in p.columns:
                            vals = p['strike'].values
                            idx = np.abs(spx_strikes_grid[None, :] - vals[:, None]).argmin(axis=1)
                            p['strike'] = spx_strikes_grid[idx]
                        elif 'strike' in p.columns:
                            p['strike'] = p['strike'].round(0).astype(int)
                    puts_list.append(p)
            except: pass

        process_component("^SPX", 1.0)
        
        # Build reference grid from SPX data
        grid_strikes = set()
        if calls_list and 'strike' in calls_list[0].columns:
            grid_strikes.update(calls_list[0]['strike'].tolist())
        if puts_list and 'strike' in puts_list[0].columns:
            grid_strikes.update(puts_list[0]['strike'].tolist())
            
        if grid_strikes:
            spx_strikes_grid = np.array(sorted(list(grid_strikes)))
            
        process_component("SPY", spy_factor)
        process_component("QQQ", qqq_factor)
        process_component("IWM", iwm_factor)
        
        combined_calls = pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
        combined_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()
        
        return combined_calls, combined_puts

    try:
        stock = get_ticker_object(ticker)
        all_calls = []
        all_puts = []
        
        for next_exp in stock.options:
            try:
                calls, puts = fetch_options_for_date(ticker, next_exp)
                if not calls.empty:
                    all_calls.append(calls)
                if not puts.empty:
                    all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching fallback options data: {e}")
        
        if all_calls:
            combined_calls = pd.concat(all_calls, ignore_index=True)
        else:
            combined_calls = pd.DataFrame()
        if all_puts:
            combined_puts = pd.concat(all_puts, ignore_index=True)
        else:
            combined_puts = pd.DataFrame()
        
        return combined_calls, combined_puts
    except Exception as e:
        st.error(f"Error fetching all options: {e}")
        return pd.DataFrame(), pd.DataFrame()

def clear_page_state():
    """Clear all page-specific content and containers"""
    # Clear containers stored in session state
    for key in list(st.session_state.keys()):
        if key.endswith('_container') and hasattr(st.session_state[key], 'empty'):
            try:
                st.session_state[key].empty()
            except:
                pass

    for key in list(st.session_state.keys()):
        if key.startswith(('container_', 'chart_', 'table_', 'page_')):
            del st.session_state[key]
    
    if 'current_page_container' in st.session_state:
        del st.session_state['current_page_container']
    
    st.empty()

def extract_expiry_from_contract(contract_symbol):
    """
    Extracts the expiration date from an option contract symbol.
    Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date formats.
    """
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                # Parse as YYMMDD
                expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            else:
                # Parse as YYYYMMDD
                expiry_date = datetime.strptime(date_str, "%Y%m%d").date()
            return expiry_date
        except ValueError:
            return None
    return None

def add_current_price_line(fig, current_price):
    """
    Adds a dashed white line at the current price to a Plotly figure.
    For horizontal bar charts, adds a horizontal line. For other charts, adds a vertical line.
    """
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7
        )
    else:
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7,
            annotation_text=f"{current_price}",
            annotation_position="top",
            annotation=dict(
                font=dict(size=st.session_state.chart_text_size)
            )
        )
    return fig

@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        # The ticker column is usually 'Symbol'
        tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
        print(f"Successfully fetched {len(tickers)} S&P 500 tickers")
        return set(tickers)
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Return a fallback empty set if fetch fails
        return set()

def get_screener_data(screener_type, filter_sp500=False):
    """Fetch screener data from Yahoo Finance, optionally filtering for S&P 500 stocks only"""
    try:
        # Request more results when filtering for S&P 500 to ensure we get enough matches
        count = 250 if filter_sp500 else 25
        response = yf.screen(screener_type, count=count)
        if isinstance(response, dict) and 'quotes' in response:
            # Get S&P 500 tickers if filtering is enabled
            sp500_tickers = get_sp500_tickers() if filter_sp500 else None
            
            data = []
            for quote in response['quotes']:
                symbol = quote.get('symbol', '')
                # Skip if filtering for S&P 500 and symbol not in list
                if filter_sp500 and sp500_tickers and symbol not in sp500_tickers:
                    continue
                # Extract relevant information
                info = {
                    'symbol': symbol,
                    'shortName': quote.get('shortName', ''),
                    'regularMarketPrice': quote.get('regularMarketPrice', 0),
                    'regularMarketChangePercent': quote.get('regularMarketChangePercent', 0),
                    'regularMarketVolume': quote.get('regularMarketVolume', 0),
                }
                data.append(info)
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching screener data: {e}")
        return pd.DataFrame()

def calculate_annualized_return(data, period='1y'):
    """Calculate annualized return rate for each weekday"""
    # Convert period to days
    period_days = {
        '1y': 365,
        '6mo': 180,
        '3mo': 90,
        '1mo': 30,
    }
    days = period_days.get(period, 365)
    
    # Filter data for selected period using proper indexing
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=days)
    filtered_data = data.loc[start_date:end_date].copy()
    
    # Calculate daily returns
    filtered_data['Returns'] = filtered_data['Close'].pct_change()
    
    # Group by weekday and calculate mean return
    weekday_returns = filtered_data.groupby(filtered_data.index.weekday)['Returns'].mean()
    
    # Annualize returns (252 trading days per year)
    annualized_returns = (1 + weekday_returns) ** 252 - 1
    
    # Map weekday numbers to names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    annualized_returns.index = weekday_names
    
    return annualized_returns.fillna(0) * 100  # Convert to percentage

def create_weekday_returns_chart(returns):
    """Create a bar chart of weekday returns"""
    fig = go.Figure()
    
    # Add bars with colors based on return value
    for day, value in returns.items():
        color = st.session_state.call_color if value >= 0 else st.session_state.put_color
        fig.add_trace(go.Bar(
            x=[day],
            y=[value],
            name=day,
            marker_color=color,
            text=[f'{value:.2f}%'],
            textposition='outside'
        ))
    
    # Calculate y-axis range with padding
    y_values = returns.values
    y_max = max(y_values)
    y_min = min(y_values)
    y_range = y_max - y_min
    padding = y_range * 0.2  # 20% padding
    
    fig.update_layout(
        title=dict(
            text='Annualized Return Rate by Weekday',
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text='Weekday',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Annualized Return (%)',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            range=[y_min - padding, y_max + padding],  # Add padding to y-axis range
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        showlegend=False,
        template="plotly_dark"
    )
    
    # Update axis fonts
    fig.update_xaxes(tickfont=dict(size=st.session_state.chart_text_size))
    
    return fig

def analyze_options_flow(calls_df, puts_df, current_price):
    """Analyze options flow focusing on Volume and Premium distribution (ITM vs OTM)."""
    # Deep copy to avoid modifying originals
    calls = calls_df.copy()
    puts = puts_df.copy()
    
    # Add ITM/OTM classification
    calls['moneyness'] = calls.apply(lambda x: 'ITM' if x['strike'] <= current_price else 'OTM', axis=1)
    puts['moneyness'] = puts.apply(lambda x: 'ITM' if x['strike'] >= current_price else 'OTM', axis=1)
    
    # Calculate Volume and Premium
    # Use Mid Price for premium calculation
    calls['midPrice'] = (calls['bid'].fillna(0) + calls['ask'].fillna(0)) / 2
    puts['midPrice'] = (puts['bid'].fillna(0) + puts['ask'].fillna(0)) / 2

    calls['premium_val'] = calls['volume'] * calls['midPrice'] * 100
    puts['premium_val'] = puts['volume'] * puts['midPrice'] * 100
    
    # Stats
    call_stats = {
        'volume': calls['volume'].sum(),
        'premium': calls['premium_val'].sum(),
        'ITM': {
            'volume': calls[calls['moneyness'] == 'ITM']['volume'].sum(),
            'premium': calls[calls['moneyness'] == 'ITM']['premium_val'].sum()
        },
        'OTM': {
            'volume': calls[calls['moneyness'] == 'OTM']['volume'].sum(),
            'premium': calls[calls['moneyness'] == 'OTM']['premium_val'].sum()
        }
    }
    
    put_stats = {
        'volume': puts['volume'].sum(),
        'premium': puts['premium_val'].sum(),
        'ITM': {
            'volume': puts[puts['moneyness'] == 'ITM']['volume'].sum(),
            'premium': puts[puts['moneyness'] == 'ITM']['premium_val'].sum()
        },
        'OTM': {
            'volume': puts[puts['moneyness'] == 'OTM']['volume'].sum(),
            'premium': puts[puts['moneyness'] == 'OTM']['premium_val'].sum()
        }
    }
    
    return {
        'calls': call_stats,
        'puts': put_stats,
        'total_premium': {
            'calls': call_stats['premium'],
            'puts': put_stats['premium']
        }
    }

def create_option_flow_charts(flow_data, title="Options Flow Analysis"):
    """Create visual charts for options flow analysis"""
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    # Chart 1: Total Volume Comparison (Call vs Put)
    fig_volume = go.Figure()
    
    fig_volume.add_trace(go.Bar(
        x=['Calls', 'Puts'],
        y=[flow_data['calls']['volume'], flow_data['puts']['volume']],
        name='Volume',
        marker_color=[call_color, put_color]
    ))
    
    fig_volume.update_layout(
        title=dict(
            text="Total Contract Volume",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(text='Option Type', font=dict(size=st.session_state.chart_text_size)),
        yaxis_title=dict(text='Volume', font=dict(size=st.session_state.chart_text_size)),
        template="plotly_dark",
        showlegend=False
    )
    
    # Chart 2: Total Premium Traded (Call vs Put)
    fig_premium = go.Figure()
    
    fig_premium.add_trace(go.Bar(
        x=['Call Premium', 'Put Premium'],
        y=[flow_data['total_premium']['calls'], flow_data['total_premium']['puts']],
        name='Premium',
        marker_color=[call_color, put_color],
        texttemplate="$%{y:,.0f}",
        textposition='auto'
    ))
    
    fig_premium.update_layout(
        title=dict(
            text="Total Premium Traded ($)",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(text='Option Type', font=dict(size=st.session_state.chart_text_size)),
        yaxis_title=dict(text='Premium Value ($)', font=dict(size=st.session_state.chart_text_size)),
        template="plotly_dark",
        showlegend=False
    )
    
    # Chart 3: ITM vs OTM Volume Details
    fig_itm_otm_vol = go.Figure()
    
    fig_itm_otm_vol.add_trace(go.Bar(
        name='ITM',
        x=['Calls', 'Puts'],
        y=[flow_data['calls']['ITM']['volume'], flow_data['puts']['ITM']['volume']],
        marker_color=[call_color, put_color],
        opacity=1.0  # Solid for ITM
    ))
    
    fig_itm_otm_vol.add_trace(go.Bar(
        name='OTM',
        x=['Calls', 'Puts'],
        y=[flow_data['calls']['OTM']['volume'], flow_data['puts']['OTM']['volume']],
        marker_color=[call_color, put_color],
        opacity=0.5  # Faded for OTM
    ))
    
    fig_itm_otm_vol.update_layout(
        title=dict(
            text="Volume Breakdown: ITM vs OTM",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        barmode='stack',
        template="plotly_dark",
        legend=dict(title="Moneyness")
    )

    # Chart 4: ITM vs OTM Premium Details
    fig_itm_otm_prem = go.Figure()
    
    fig_itm_otm_prem.add_trace(go.Bar(
        name='ITM',
        x=['Calls', 'Puts'],
        y=[flow_data['calls']['ITM']['premium'], flow_data['puts']['ITM']['premium']],
        marker_color=[call_color, put_color],
        opacity=1.0
    ))
    
    fig_itm_otm_prem.add_trace(go.Bar(
        name='OTM',
        x=['Calls', 'Puts'],
        y=[flow_data['calls']['OTM']['premium'], flow_data['puts']['OTM']['premium']],
        marker_color=[call_color, put_color],
        opacity=0.5
    ))
    
    fig_itm_otm_prem.update_layout(
        title=dict(
            text="Premium Breakdown: ITM vs OTM ($)",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        barmode='stack',
        template="plotly_dark",
        legend=dict(title="Moneyness"),
        yaxis=dict(tickformat="$,.0f")
    )
    
    return fig_volume, fig_premium, fig_itm_otm_vol, fig_itm_otm_prem

def create_option_premium_heatmap(calls_df, puts_df, strikes, expiry_dates, current_price):
    """Create a heatmap showing premium distribution across strikes and expiries"""
    # Initialize data matrices
    call_premium = np.zeros((len(expiry_dates), len(strikes)))
    put_premium = np.zeros((len(expiry_dates), len(strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            vol = row['volume'] if pd.notna(row['volume']) else 0
            bid = row['bid'] if pd.notna(row['bid']) else 0
            ask = row['ask'] if pd.notna(row['ask']) else 0
            price = (bid + ask) / 2
            call_premium[i, j] = vol * price * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            vol = row['volume'] if pd.notna(row['volume']) else 0
            bid = row['bid'] if pd.notna(row['bid']) else 0
            ask = row['ask'] if pd.notna(row['ask']) else 0
            price = (bid + ask) / 2
            put_premium[i, j] = vol * price * 100
    
    # Create heatmaps
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(call_color[1:3], 16)},{int(call_color[3:5], 16)},{int(call_color[5:7], 16)},0.1)'], [1, call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Premium ($)",
                side="top",
                font=dict(color="white")
            ),
            tickfont=dict(color="white"),
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(put_color[1:3], 16)},{int(put_color[3:5], 16)},{int(put_color[5:7], 16)},0.1)'], [1, put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Premium ($)",
                side="top",
                font=dict(color="white")
            ),
            tickfont=dict(color="white"),
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

def create_premium_heatmap(calls_df, puts_df, filtered_strikes, selected_expiry_dates, current_price):
    """Create heatmaps showing premium distribution across strikes and expiration dates."""
    # Initialize data matrices
    call_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    put_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(filtered_strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(selected_expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            vol = row['volume'] if pd.notna(row['volume']) else 0
            bid = row['bid'] if pd.notna(row['bid']) else 0
            ask = row['ask'] if pd.notna(row['ask']) else 0
            price = (bid + ask) / 2
            call_premium[expiry_idx][strike_idx] += vol * price * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            vol = row['volume'] if pd.notna(row['volume']) else 0
            bid = row['bid'] if pd.notna(row['bid']) else 0
            ask = row['ask'] if pd.notna(row['ask']) else 0
            price = (bid + ask) / 2
            put_premium[expiry_idx][strike_idx] += vol * price * 100
    
    # Create heatmaps
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(call_color[1:3], 16)},{int(call_color[3:5], 16)},{int(call_color[5:7], 16)},0.1)'], [1, call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title=dict(text="Premium ($)", font=dict(color="white")),
            tickfont=dict(color="white"),
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(put_color[1:3], 16)},{int(put_color[3:5], 16)},{int(put_color[5:7], 16)},0.1)'], [1, put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title=dict(text="Premium ($)", font=dict(color="white")),
            tickfont=dict(color="white"),
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

# Removed: def create_premium_ratio_chart(calls_df, puts_df): function is deleted

# -------------------------------
# Fetch all options experations and add extract expiry
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker in parallel.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
    print(f"Fetching avaiable expirations for {ticker}")  # Add print statement
    stock = get_ticker_object(ticker)
    all_calls = []
    all_puts = []
    
    if stock.options:
        # Get current market date
        current_market_date = get_now_et().date()
        
        # Filter for valid future expirations first
        valid_expirations = []
        for exp in stock.options:
            try:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                if exp_date >= current_market_date:
                    valid_expirations.append(exp)
            except ValueError:
                continue

        # Capture context for threads
        ctx = get_script_run_ctx() if get_script_run_ctx else None

        def fetch_single_expiry(exp):
            if add_script_run_ctx and ctx:
                add_script_run_ctx(threading.current_thread(), ctx)
            try:
                # Use fetch_options_for_date to leverage its caching
                calls, puts = fetch_options_for_date(ticker, exp)
                return calls, puts
            except Exception as e:
                st.error(f"Error fetching chain for expiry {exp}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_exp = {}
            for exp in valid_expirations:
                future = executor.submit(fetch_single_expiry, exp)
                future_to_exp[future] = exp
            
            for future in concurrent.futures.as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    if result:
                        calls, puts = result
                        if not calls.empty:
                            all_calls.append(calls)
                        if not puts.empty:
                            all_puts.append(puts)
                except Exception as e:
                    print(f"Exception fetching expiry {exp}: {e}")

    else:
        try:
            # Get next valid expiration
            next_exp = stock.options[0] if stock.options else None
            if next_exp:
                calls, puts = fetch_options_for_date(ticker, next_exp)
                if not calls.empty:
                    all_calls.append(calls)
                if not puts.empty:
                    all_puts.append(puts)
        except Exception as e:
            st.error(f"Error fetching fallback options data: {e}")
    
    if all_calls:
        combined_calls = pd.concat(all_calls, ignore_index=True)
    else:
        combined_calls = pd.DataFrame()
    if all_puts:
        combined_puts = pd.concat(all_puts, ignore_index=True)
    else:
        combined_puts = pd.DataFrame()
    
    return combined_calls, combined_puts

# Charts and price fetching
def get_ticker_object(ticker):
    """Helper to get yf.Ticker object, handling MARKET special case"""
    if ticker == "MARKET":
        return yf.Ticker("^SPX")
    return yf.Ticker(ticker)

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)  # Cache TTL matches refresh rate
def get_current_price(ticker):
    """Get current price with fallback logic"""
    if ticker == "MARKET":
        return get_current_price("^SPX")

    print(f"Fetching current price for {ticker}")
    formatted_ticker = ticker.replace('%5E', '^')
    
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                return round(float(price), 2)
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
    
    try:
        stock = get_ticker_object(ticker)
        price = stock.info.get("regularMarketPrice")
        if price is None:
            price = stock.fast_info.get("lastPrice")
        if price is not None:
            return round(float(price), 2)
    except Exception as e:
        print(f"Yahoo Finance error: {str(e)}")
    
    return None

def create_oi_volume_charts(calls, puts, S, date_count=1):
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for OI and Volume, filtering out zeros
    calls_oi_df = calls_filtered[['strike', 'openInterest']].copy().fillna(0)
    calls_oi_df = calls_oi_df[calls_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    # Aggregate by strike to handle multiple expirations
    calls_oi_df = calls_oi_df.groupby('strike', as_index=False)['openInterest'].sum()
    calls_oi_df['OptionType'] = 'Call'
    
    puts_oi_df = puts_filtered[['strike', 'openInterest']].copy().fillna(0)
    puts_oi_df = puts_oi_df[puts_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    # Aggregate by strike to handle multiple expirations
    puts_oi_df = puts_oi_df.groupby('strike', as_index=False)['openInterest'].sum()
    puts_oi_df['OptionType'] = 'Put'
    
    calls_vol_df = calls_filtered[['strike', 'volume']].copy().fillna(0)
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    # Aggregate by strike to handle multiple expirations
    calls_vol_df = calls_vol_df.groupby('strike', as_index=False)['volume'].sum()
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy().fillna(0)
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    # Aggregate by strike to handle multiple expirations
    puts_vol_df = puts_vol_df.groupby('strike', as_index=False)['volume'].sum()
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Open Interest and Net Volume using filtered data
    net_oi = calls_filtered.groupby('strike')['openInterest'].sum().sub(puts_filtered.groupby('strike')['openInterest'].sum(), fill_value=0)
    net_volume = calls_filtered.groupby('strike')['volume'].sum().sub(puts_filtered.groupby('strike')['volume'].sum(), fill_value=0)
    
    # Calculate total values for titles using the entire chain
    total_call_oi = calls['openInterest'].sum()
    total_put_oi = puts['openInterest'].sum()
    total_net_oi = total_call_oi - total_put_oi
    
    total_call_volume = calls['volume'].sum()
    total_put_volume = puts['volume'].sum()
    total_net_volume = total_call_volume - total_put_volume
    
    date_suffix = f" ({date_count} dates)" if date_count > 1 else ""

    # Determine colors for net values
    net_oi_color = call_color if total_net_oi >= 0 else put_color
    net_volume_color = call_color if total_net_volume >= 0 else put_color

    # Create titles with totals using HTML for colored values
    oi_title_with_totals = (
        f"Open Interest by Strike{date_suffix}     "
        f"<span style='color: {call_color}'>{format_large_number(total_call_oi)}</span> | "
        f"<span style='color: {net_oi_color}'>Net: {format_large_number(total_net_oi)}</span> | "
        f"<span style='color: {put_color}'>{format_large_number(total_put_oi)}</span>"
    )
    
    volume_title_with_totals = (
        f"Volume by Strike{date_suffix}     "
        f"<span style='color: {call_color}'>{format_large_number(total_call_volume)}</span> | "
        f"<span style='color: {net_volume_color}'>Net: {format_large_number(total_net_volume)}</span> | "
        f"<span style='color: {put_color}'>{format_large_number(total_put_volume)}</span>"
    )
    
    # Create Open Interest Chart
    fig_oi = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=calls_oi_df['strike'],
                x=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=puts_oi_df['strike'],
                x=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net OI if enabled
    if st.session_state.show_net and not net_oi.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=net_oi.index,
                y=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=net_oi.index,
                x=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for OI chart
    y_values = []
    for trace in fig_oi.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        oi_y_min = min(min(y_values), 0)  # Include 0 in the range
        oi_y_max = max(y_values)
        oi_y_range = oi_y_max - oi_y_min
        
        # Add 15% padding on top and 5% on bottom
        oi_padding_top = oi_y_range * 0.15
        oi_padding_bottom = oi_y_range * 0.05
        oi_y_min = oi_y_min - oi_padding_bottom
        oi_y_max = oi_y_max + oi_padding_top
    else:
        # Default values if no valid y values
        oi_y_min = 0
        oi_y_max = 100
    
    # Add padding for x-axis range
    padding = strike_range * 0.1
    
    # Update OI chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for volume chart
    y_values = []
    for trace in fig_volume.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        vol_y_min = min(min(y_values), 0)  # Include 0 in the range
        vol_y_max = max(y_values)
        vol_y_range = vol_y_max - vol_y_min
        
        # Add 15% padding on top and 5% on bottom
        vol_padding_top = vol_y_range * 0.15
        vol_padding_bottom = vol_y_range * 0.05
        vol_y_min = vol_y_min - vol_padding_bottom
        vol_y_max = vol_y_max + vol_padding_top
    else:
        # Default values if no valid y values
        vol_y_min = 0
        vol_y_max = 100
    
    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_oi, fig_volume

def create_volume_by_strike_chart(calls, puts, S, date_count=1):
    """Create a standalone volume by strike chart for the dashboard."""
    if S is None:
        st.error("Could not fetch underlying price.")
        return None

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for Volume, filtering out zeros
    calls_vol_df = calls_filtered[['strike', 'volume']].copy().fillna(0)
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]
    # Aggregate by strike to handle multiple expirations
    calls_vol_df = calls_vol_df.groupby('strike', as_index=False)['volume'].sum()
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy().fillna(0)
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]
    # Aggregate by strike to handle multiple expirations
    puts_vol_df = puts_vol_df.groupby('strike', as_index=False)['volume'].sum()
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Volume using filtered data
    net_volume = calls_filtered.groupby('strike')['volume'].sum().sub(puts_filtered.groupby('strike')['volume'].sum(), fill_value=0)
    
    # Calculate total values for title using the entire chain
    total_call_volume = calls['volume'].sum()
    total_put_volume = puts['volume'].sum()
    total_net_volume = total_call_volume - total_put_volume
    
    date_suffix = f" ({date_count} dates)" if date_count > 1 else ""

    # Determine color for net value
    net_volume_color = call_color if total_net_volume >= 0 else put_color

    # Create title with totals using HTML for colored values
    volume_title_with_totals = (
        f"Volume by Strike{date_suffix}     "
        f"<span style='color: {call_color}'>{format_large_number(total_call_volume)}</span> | "
        f"<span style='color: {net_volume_color}'>Net: {format_large_number(total_net_volume)}</span> | "
        f"<span style='color: {put_color}'>{format_large_number(total_put_volume)}</span>"
    )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    
    # Add current price line
    S = round(S, 2)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_volume

def create_donut_chart(call_volume, put_volume, date_count=1):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    # Get colors directly from session state at creation time
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    date_suffix = f" ({date_count} dates)" if date_count > 1 else ""

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text=f'Call vs Put Volume Ratio{date_suffix}',
        title_font_size=st.session_state.chart_text_size + 3,  # Title slightly larger
        showlegend=True,
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        )
    )
    fig.update_traces(
        hoverinfo='label+percent+value',
        marker=dict(colors=[call_color, put_color]),
        textfont=dict(size=st.session_state.chart_text_size)
    )
    return fig

# Greek Calculations
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_risk_free_rate():
    """Fetch the current risk-free rate from the 3-month Treasury Bill yield with caching."""
    try:
        # Get current price for the 3-month Treasury Bill
        irx_rate = get_current_price("^IRX")
        
        if irx_rate is not None:
            # Convert percentage to decimal (e.g., 5.2% to 0.052)
            risk_free_rate = irx_rate / 100
        else:
            # Fallback to a default value if price fetch fails
            risk_free_rate = 0.02  # 2% as fallback
            print("Using fallback risk-free rate of 2%")
            
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.02  # 2% as fallback

# Initialize risk-free rate in session state if not already present
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = get_risk_free_rate()

def calculate_bs_price(flag, S, K, t, r, sigma, q=0):
    """Calculate Black-Scholes option price with dividends."""
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        if flag == 'c':
            price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
        return price
    except:
        return 0.0

def calculate_bs_vega(S, K, t, r, sigma, q=0):
    """Calculate Black-Scholes Vega with dividends."""
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        return S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
    except:
        return 0.0

def calculate_implied_volatility(price, S, K, t, r, flag, q=0):
    """Calculate Implied Volatility using Newton-Raphson method."""
    sigma = 0.5  # Initial guess
    for i in range(100):
        bs_price = calculate_bs_price(flag, S, K, t, r, sigma, q)
        diff = price - bs_price
        
        if abs(diff) < 1e-5:
            return sigma
            
        vega = calculate_bs_vega(S, K, t, r, sigma, q)
        if abs(vega) < 1e-8:
            return None
            
        sigma = sigma + diff / vega
        
        if sigma <= 0:
            sigma = 0.001 # Reset if negative
        if sigma > 5:
            sigma = 5.0 # Cap if too high
            
    return None

def calculate_greeks(flag, S, K, t, sigma, r=None, q=0):
    """
    Calculate delta, gamma and vanna for an option using Black-Scholes model with dividends.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1e-5)  # Minimum ~5 minutes expressed in years
        if r is None:
            r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Calculate delta
        if flag == 'c':
            delta_val = np.exp(-q * t) * norm.cdf(d1)
        else:  # put
            delta_val = np.exp(-q * t) * (norm.cdf(d1) - 1)
        
        # Calculate gamma
        gamma_val = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
        
        # Calculate vega
        vega_val = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
        
        # Calculate vanna
        vanna_val = -np.exp(-q * t) * norm.pdf(d1) * d2 / sigma
        
        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

def calculate_charm(flag, S, K, t, sigma, r=None, q=0):
    """
    Calculate charm (dDelta/dTime) for an option with dividends.
    """
    try:
        t = max(t, 1e-5)
        if r is None:
            r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        norm_d1 = norm.pdf(d1)
        
        if flag == 'c':
            charm = -np.exp(-q * t) * (norm_d1 * (2*(r-q)*t - d2*sigma*np.sqrt(t)) / (2*t*sigma*np.sqrt(t)) - q * norm.cdf(d1))
        else:
            charm = -np.exp(-q * t) * (norm_d1 * (2*(r-q)*t - d2*sigma*np.sqrt(t)) / (2*t*sigma*np.sqrt(t)) + q * norm.cdf(-d1))
        
        return charm
    except Exception as e:
        st.error(f"Error calculating charm: {e}")
        return None

def calculate_speed(flag, S, K, t, sigma, r=None, q=0):
    """
    Calculate speed (dGamma/dSpot) for an option with dividends.
    """
    try:
        t = max(t, 1e-5)
        if r is None:
            r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Calculate gamma manually
        gamma = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
        
        # Calculate speed
        speed = -gamma * (d1/(sigma * np.sqrt(t)) + 1) / S
        
        return speed
    except Exception as e:
        st.error(f"Error calculating speed: {e}")
        return None

def calculate_vomma(flag, S, K, t, sigma, r=None, q=0):
    """
    Calculate vomma (dVega/dVol) for an option with dividends.
    """
    try:
        t = max(t, 1e-5)
        if r is None:
            r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Calculate vega manually
        vega = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
        
        # Calculate vomma
        vomma = vega * (d1 * d2) / sigma
        
        return vomma
    except Exception as e:
        st.error(f"Error calculating vomma: {e}")
        return None

def calculate_color(flag, S, K, t, sigma, r=None, q=0):
    """
    Calculate Color (dGamma/dTime) for an option.
    Color measures the rate of change of Gamma over time.
    """
    try:
        t = max(t, 1e-5)
        if r is None:
            r = st.session_state.risk_free_rate

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        norm_d1 = norm.pdf(d1)
        
        # Color Calculation
        term1 = 2 * (r - q) * t
        term2 = d2 * sigma * np.sqrt(t)
        
        # Using a standard formula for Color
        # Color = -exp(-qt) * [N'(d1) / (2*S*t*sigma*sqrt(t))] * [1 + (2(r-q)t - d2*sigma*sqrt(t)) * d1 / (2*t*sigma*sqrt(t))]
        # Simplified:
        color = -np.exp(-q*t) * (norm_d1 / (2 * S * t * sigma * np.sqrt(t))) * \
                (1 + (term1 - term2) * d1 / (2 * t * sigma * np.sqrt(t)))
        
        return color
    except Exception as e:
        return None

def calculate_implied_move(S, calls_df, puts_df):
    """Calculate implied move based on straddle prices."""
    try:
        # Find ATM strike (closest to current price)
        all_strikes = pd.concat([calls_df['strike'], puts_df['strike']]).unique()
        atm_strike = min(all_strikes, key=lambda x: abs(x - S))
        
        # Get ATM call and put prices
        atm_call = calls_df[calls_df['strike'] == atm_strike]
        atm_put = puts_df[puts_df['strike'] == atm_strike]
        
        if not atm_call.empty and not atm_put.empty:
            # Use Mid Price for better accuracy
            call_bid = atm_call['bid'].iloc[0]
            call_ask = atm_call['ask'].iloc[0]
            call_price = (call_bid + call_ask) / 2

            put_bid = atm_put['bid'].iloc[0]
            put_ask = atm_put['ask'].iloc[0]
            put_price = (put_bid + put_ask) / 2
            
            straddle_price = call_price + put_price
            implied_move_pct = (straddle_price / S) * 100
            implied_move_dollars = straddle_price
            
            return {
                'atm_strike': atm_strike,
                'straddle_price': straddle_price,
                'implied_move_pct': implied_move_pct,
                'implied_move_dollars': implied_move_dollars,
                'upper_range': S + implied_move_dollars,
                'lower_range': S - implied_move_dollars
            }
    except Exception as e:
        print(f"Error calculating implied move: {e}")
    
    return None

def find_probability_strikes(calls_df, puts_df, S, expiry_date, target_prob=0.5, q=0):
    """Find strikes where there's exactly target_prob chance of being above/below at expiration."""
    try:
        # Calculate probability distribution first
        prob_df = calculate_probability_distribution(calls_df, puts_df, S, expiry_date, q)
        
        if prob_df.empty:
            return None
        
        # For 50% probability: find the strike where prob_above ≈ 0.5 (median)
        # For other probabilities: find strikes where prob_above = target_prob and prob_above = 1-target_prob
        
        # Industry standard: Use target_prob directly as delta levels
        # For 16 delta: find strikes where prob_above = 0.16 and prob_above = 0.84
        # For 30 delta: find strikes where prob_above = 0.30 and prob_above = 0.70
        
        # Lower bound: target_prob chance of being above
        lower_prob = target_prob
        # Upper bound: (1-target_prob) chance of being above  
        upper_prob = 1 - target_prob
        
        # Find strike where prob_above = lower_prob (lower bound of confidence interval)
        prob_above_target = prob_df.iloc[(prob_df['prob_above'] - lower_prob).abs().argsort()[:1]]
        strike_above = prob_above_target['strike'].iloc[0] if not prob_above_target.empty else None
        actual_prob_above = prob_above_target['prob_above'].iloc[0] if not prob_above_target.empty else None
        
        # Find strike where prob_above = upper_prob (upper bound of confidence interval)
        prob_below_target = prob_df.iloc[(prob_df['prob_above'] - upper_prob).abs().argsort()[:1]]
        strike_below = prob_below_target['strike'].iloc[0] if not prob_below_target.empty else None
        actual_prob_below = 1 - prob_below_target['prob_above'].iloc[0] if not prob_below_target.empty else None
        
        return {
            'strike_above': strike_above,  # Strike with target_prob chance of being above
            'prob_above': actual_prob_above,
            'strike_below': strike_below,  # Strike with target_prob chance of being below  
            'prob_below': actual_prob_below,
            'target_probability': target_prob
        }
    except Exception as e:
        print(f"Error finding probability strikes: {e}")
        return None

def find_delta_strikes(calls_df, puts_df, target_delta=0.5):
    """Find strikes closest to the target delta (for delta-based analysis)."""
    try:
        # For calls, find strike closest to target delta
        if 'calc_delta' in calls_df.columns:
            call_deltas = calls_df[calls_df['calc_delta'].notna()]
            if not call_deltas.empty:
                call_target = call_deltas.iloc[(call_deltas['calc_delta'] - target_delta).abs().argsort()[:1]]
                call_strike = call_target['strike'].iloc[0] if not call_target.empty else None
                call_delta = call_target['calc_delta'].iloc[0] if not call_target.empty else None
            else:
                call_strike, call_delta = None, None
        else:
            call_strike, call_delta = None, None
        
        # For puts, find strike closest to -target_delta (puts have negative delta)
        if 'calc_delta' in puts_df.columns:
            put_deltas = puts_df[puts_df['calc_delta'].notna()]
            if not put_deltas.empty:
                put_target = put_deltas.iloc[(put_deltas['calc_delta'] - (-target_delta)).abs().argsort()[:1]]
                put_strike = put_target['strike'].iloc[0] if not put_target.empty else None
                put_delta = put_target['calc_delta'].iloc[0] if not put_target.empty else None
            else:
                put_strike, put_delta = None, None
        else:
            put_strike, put_delta = None, None
        
        return {
            'call_strike': call_strike,
            'call_delta': call_delta,
            'put_strike': put_strike,
            'put_delta': put_delta,
            'call_prob_itm': call_delta if call_delta else None,
            'put_prob_itm': abs(put_delta) if put_delta else None
        }
    except Exception as e:
        print(f"Error finding delta strikes: {e}")
        return None

def calculate_probability_distribution(calls_df, puts_df, S, expiry_date, q=0):
    """Calculate probability distribution from option prices using risk-neutral probabilities."""
    try:
        # Get all strikes and sort them
        all_strikes = sorted(pd.concat([calls_df['strike'], puts_df['strike']]).unique())
        
        probabilities = []
        strikes_data = []
        
        today = get_now_et().date()
        if isinstance(expiry_date, str):
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        
        # Calculate time to expiration more precisely
        t = calculate_time_to_expiration(expiry_date)
        t = max(t, 1e-5)  # Ensure positive
        r = st.session_state.risk_free_rate
        
        for strike in all_strikes:
            # Get call and put data for this strike
            call_data = calls_df[calls_df['strike'] == strike]
            put_data = puts_df[puts_df['strike'] == strike]
            
            # Calculate IV manually using Mid Price
            iv = None
            try:
                # Use Call data if available, else Put
                if not call_data.empty:
                    row = call_data.iloc[0]
                    flag = 'c'
                elif not put_data.empty:
                    row = put_data.iloc[0]
                    flag = 'p'
                else:
                    continue

                bid = row.get('bid', 0)
                ask = row.get('ask', 0)
                price = (bid + ask) / 2
                
                if price > 0:
                    iv = calculate_implied_volatility(price, S, strike, t, r, flag, q)
            except:
                pass
            
            if iv and iv > 0 and iv <= 5.0:
                # Calculate risk-neutral probability using Black-Scholes
                try:
                    d1 = (log(S / strike) + (r - q + 0.5 * iv**2) * t) / (iv * sqrt(t))
                    d2 = d1 - iv * sqrt(t)
                    
                    # Risk-neutral probability of finishing above strike
                    prob_above = norm.cdf(d2)  # Use d2 for risk-neutral probability
                    
                    probabilities.append(prob_above)
                    strikes_data.append(strike)
                except:
                    continue
            else:
                # Fallback to delta if available
                if not call_data.empty and 'calc_delta' in call_data.columns:
                    delta = call_data['calc_delta'].iloc[0]
                    prob_above = delta
                elif not put_data.empty and 'calc_delta' in put_data.columns:
                    delta = put_data['calc_delta'].iloc[0]
                    prob_above = 1 + delta  # Put delta is negative
                else:
                    continue
                
                probabilities.append(prob_above)
                strikes_data.append(strike)
        
        if not strikes_data:
            return pd.DataFrame()
            
        # Sort by strike
        prob_df = pd.DataFrame({
            'strike': strikes_data,
            'prob_above': probabilities,
            'prob_below': [1 - p for p in probabilities]
        }).sort_values('strike').reset_index(drop=True)
        
        return prob_df
        
    except Exception as e:
        print(f"Error calculating probability distribution: {e}")
        return pd.DataFrame()

def create_implied_probabilities_chart(prob_df, S, prob_16_data, prob_30_data, implied_move_data):
    """Create simplified implied probabilities visualization focusing on key levels."""
    try:
        # Get colors from session state
        call_color = st.session_state.call_color
        put_color = st.session_state.put_color

        # Create two simple bar charts: one for probability levels and one for expected range
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Key Probability Levels',
                'Expected Trading Range'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # First subplot: Key Probability Levels
        prob_levels = []
        
        # Add current price
        prob_levels.append({
            'level': 'Current Price',
            'strike': S,
            'type': 'neutral'
        })
        
                # Add probability levels (delta-based)
        if prob_16_data:
            if prob_16_data['strike_above']:
                prob_levels.append({
                    'level': '16Δ Above (1σ)',
                    'strike': prob_16_data['strike_above'],
                    'type': 'call'
                })
            if prob_16_data['strike_below']:
                prob_levels.append({
                    'level': '16Δ Below (1σ)',
                    'strike': prob_16_data['strike_below'],
                    'type': 'put'
                })
        
        if prob_30_data:
            if prob_30_data['strike_above']:
                prob_levels.append({
                    'level': '30Δ Above',
                    'strike': prob_30_data['strike_above'],
                    'type': 'call'
                })
            if prob_30_data['strike_below']:
                prob_levels.append({
                    'level': '30Δ Below', 
                    'strike': prob_30_data['strike_below'],
                    'type': 'put'
                })
        
        if prob_levels:
            levels_df = pd.DataFrame(prob_levels)
            # Sort by strike price for better visualization
            levels_df = levels_df.sort_values('strike')
            
            fig.add_trace(
                go.Bar(
                    x=levels_df['level'],
                    y=levels_df['strike'],
                    name='Probability Levels',
                    marker_color=[
                        call_color if row['type'] == 'call' 
                        else put_color if row['type'] == 'put'
                        else 'yellow' for _, row in levels_df.iterrows()
                    ],
                    text=[f"${v:.2f}" for v in levels_df['strike']],
                    textposition='auto',
                    textfont=dict(size=st.session_state.chart_text_size)
                ),
                row=1, col=1
            )
            fig.update_yaxes(range=[S-(0.025*S),S+(0.025*S)])
        
        # Second subplot: Expected Trading Range
        if implied_move_data:
            range_data = [
                {'level': 'Lower Range', 'value': implied_move_data['lower_range'], 'type': 'put'},
                {'level': 'Current Price', 'value': S, 'type': 'neutral'},
                {'level': 'Upper Range', 'value': implied_move_data['upper_range'], 'type': 'call'}
            ]
            range_df = pd.DataFrame(range_data)
            
            fig.add_trace(
                go.Bar(
                    x=range_df['level'],
                    y=range_df['value'],
                    name='Trading Range',
                    marker_color=[
                        put_color if row['type'] == 'put'
                        else call_color if row['type'] == 'call'
                        else 'yellow' for _, row in range_df.iterrows()
                    ],
                    text=[f"${v:.2f}" for v in range_df['value']],
                    textposition='auto',
                    textfont=dict(size=st.session_state.chart_text_size)
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=500,  # Reduced height since we have fewer elements
            title=dict(
                text="Implied Probabilities Analysis",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            showlegend=False,  # No need for legend
            template="plotly_dark",
            # Add more vertical space for labels
            margin=dict(t=100, b=50)
        )
        
        # Update subplot titles
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=st.session_state.chart_text_size + 3)
        
        # Update axes
        fig.update_xaxes(
            title_text="Probability Level",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Strike Price ($)",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Price Level",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Price ($)",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=2
        )
        
        return fig
    except Exception as e:
        print(f"Error creating implied probabilities chart: {e}")
        return go.Figure()

# Add error handling for fetching the last price to avoid KeyError.
def get_last_price(stock):
    """Helper function to get the last price of the stock."""
    return get_current_price(stock.ticker)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = get_now_et().date()
        # Allow expirations within -1 days
        days_difference = (expiry_date - current_market_date).days
        return days_difference >= -1
    except Exception:
        return False

def is_valid_trading_day(expiry_date, current_date):
    """Helper function to check if expiry is within valid trading window"""
    days_difference = (expiry_date - current_date).days
    return days_difference >= -1

def fetch_and_process_multiple_dates(ticker, expiry_dates, process_func):
    """
    Fetches and processes data for multiple expiration dates in parallel.
    
    Args:
        ticker: Stock ticker symbol
        expiry_dates: List of expiration dates
        process_func: Function to process data for each date
        
    Returns:
        Tuple of processed calls and puts DataFrames
    """
    all_calls = []
    all_puts = []
    
    # Capture context for threads
    ctx = get_script_run_ctx() if get_script_run_ctx else None

    def process_single_date(date):
        if add_script_run_ctx and ctx:
            add_script_run_ctx(threading.current_thread(), ctx)
        try:
            return process_func(ticker, date)
        except Exception as e:
            print(f"Error processing date {date}: {e}")
            return None

    # Use ThreadPoolExecutor to fetch data in parallel
    # Limit max_workers to avoid hitting API rate limits too hard, though yfinance is generally lenient
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_date = {}
        for date in expiry_dates:
            future = executor.submit(process_single_date, date)
            future_to_date[future] = date
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result = future.result()
                if result is not None:
                    calls, puts = result
                    if not calls.empty:
                        calls = calls.copy()
                        calls['expiry_date'] = date  # Add expiry date column
                        all_calls.append(calls)
                    if not puts.empty:
                        puts = puts.copy()
                        puts['expiry_date'] = date  # Add expiry date column
                        all_puts.append(puts)
            except Exception as e:
                print(f"Exception for date {date}: {e}")
    
    if all_calls and all_puts:
        combined_calls = pd.concat(all_calls, ignore_index=True)
        combined_puts = pd.concat(all_puts, ignore_index=True)
        return combined_calls, combined_puts
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)  # Cache TTL matches refresh rate
def get_combined_intraday_data(ticker):
    """Get intraday data with fallback logic"""
    formatted_ticker = ticker.replace('%5E', '^')
    
    # Use ^GSPC for SPX chart data
    chart_ticker = ticker
    if ticker == "MARKET" or formatted_ticker in ['^SPX', 'SPX', '%5ESPX']:
        chart_ticker = '^GSPC'
        
    stock = yf.Ticker(chart_ticker)
    intraday_data = stock.history(period="1d", interval="1m")
    
    # Filter for market hours (9:30 AM to 4:00 PM ET)
    if not intraday_data.empty:
        # Convert timezone to ET
        eastern = pytz.timezone('US/Eastern')
        intraday_data.index = intraday_data.index.tz_convert(eastern)
        market_start = pd.Timestamp(intraday_data.index[0].date()).replace(hour=9, minute=30)
        market_end = pd.Timestamp(intraday_data.index[0].date()).replace(hour=16, minute=0)
        intraday_data = intraday_data.between_time('09:30', '16:00')
    
    if intraday_data.empty:
        return None, None, None
    
    # Get VIXY data if overlay is enabled
    vix_data = None
    if st.session_state.show_vix_overlay:
        try:
            vix = yf.Ticker('VIXY')
            vix_intraday = vix.history(period="1d", interval="1m")
            if not vix_intraday.empty:
                # Convert timezone to ET and filter for market hours
                vix_intraday.index = vix_intraday.index.tz_convert(eastern)
                vix_data = vix_intraday.between_time('09:30', '16:00')
                
                # Align VIXY data with stock data timeframes
                if not intraday_data.empty and not vix_data.empty:
                    common_times = intraday_data.index.intersection(vix_data.index)
                    if len(common_times) > 0:
                        intraday_data = intraday_data.loc[common_times]
                        vix_data = vix_data.loc[common_times]
                    else:
                        vix_data = None
        except Exception as e:
            print(f"Error fetching VIXY data: {e}")
            vix_data = None
    
    intraday_data = intraday_data.copy()
    yahoo_last_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else None
    latest_price = yahoo_last_price
    
    # Use ^GSPC for SPX
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                latest_price = round(float(price), 2)
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
            latest_price = yahoo_last_price
    else:
        try:
            price = get_current_price(ticker)  # Use cached get_current_price
            if price is not None:
                latest_price = round(float(price), 2)
                last_idx = intraday_data.index[-1]
                new_idx = last_idx + pd.Timedelta(minutes=1)
                new_row = pd.DataFrame({
                    'Open': [latest_price],
                    'High': [latest_price],
                    'Low': [latest_price],
                    'Close': [latest_price],
                    'Volume': [0]
                }, index=[new_idx])
                intraday_data = pd.concat([intraday_data, new_row])
        except Exception as e:
            print(f"Error updating latest price: {str(e)}")
    
    return intraday_data, latest_price, vix_data

def create_iv_surface(calls_df, puts_df, current_price, selected_dates=None):
    """Create data for IV surface plot with enhanced smoothing and data validation."""
    # Filter by selected dates if provided
    if selected_dates:
        calls_df = calls_df[calls_df['extracted_expiry'].isin(selected_dates)]
        puts_df = puts_df[puts_df['extracted_expiry'].isin(selected_dates)]
    
    # Add flag for IV calculation
    calls_df = calls_df.copy()
    puts_df = puts_df.copy()
    calls_df['flag'] = 'c'
    puts_df['flag'] = 'p'

    # Combine calls and puts
    options_data = pd.concat([calls_df, puts_df])
    
    # Calculate IV manually
    r = st.session_state.get('risk_free_rate', 0.04)
    
    # Pre-calculate time to expiration for all unique dates to avoid repeated calls
    unique_dates = options_data['extracted_expiry'].unique()
    t_map = {date: max(calculate_time_to_expiration(date), 1e-5) for date in unique_dates}

    def calc_iv_safe(row):
        try:
            t = t_map.get(row['extracted_expiry'])
            if t is None: return None
            
            bid = row.get('bid', 0)
            ask = row.get('ask', 0)
            price = (bid + ask) / 2
            
            if price <= 0: return None
            
            iv = calculate_implied_volatility(price, current_price, row['strike'], t, r, row['flag'])
            if iv is not None and 0 < iv <= 5.0:
                return iv
            return None
        except:
            return None

    options_data['calc_iv'] = options_data.apply(calc_iv_safe, axis=1)
    
    # Drop rows with invalid calculated IV
    options_data = options_data.dropna(subset=['calc_iv', 'strike', 'extracted_expiry'])
    
    if options_data.empty:
        st.warning("No valid options data available for IV surface.")
        return None, None, None
    
    # Calculate moneyness and months to expiration
    options_data['moneyness'] = options_data['strike'].apply(
        lambda x: (x / current_price) * 100
    )
    
    options_data['months'] = options_data['extracted_expiry'].apply(
        lambda x: (x - get_now_et().date()).days / 30.44
    )
    
    # Remove extreme values (using calc_iv instead of impliedVolatility)
    for col in ['calc_iv', 'moneyness', 'months']:
        q1 = options_data[col].quantile(0.01)
        q99 = options_data[col].quantile(0.99)
        options_data = options_data[
            (options_data[col] >= q1) & 
            (options_data[col] <= q99)
        ]
    
    if options_data.empty:
        st.warning("No valid data points after filtering.")
        return None, None, None
    
    # Create grid for interpolation
    moneyness_range = np.linspace(85, 115, 200)
    months_range = np.linspace(
        options_data['months'].min(),
        options_data['months'].max(),
        200
    )
    
    # Create meshgrid
    X, Y = np.meshgrid(moneyness_range, months_range)
    
    try:
        # Prepare data for interpolation
        points = options_data[['moneyness', 'months']].values
        values = options_data['calc_iv'].values * 100  # Use calc_iv
        
        # Initial interpolation
        Z = griddata(
            points,
            values,
            (X, Y),
            method='linear',  # Start with linear interpolation
            fill_value=np.nan
        )
        
        # Fill remaining NaN values with nearest neighbor interpolation
        mask = np.isnan(Z)
        Z[mask] = griddata(
            points,
            values,
            (X[mask], Y[mask]),
            method='nearest'
        )
        
        # Apply Gaussian smoothing with multiple passes
        if not np.isnan(Z).any():  # Only smooth if we have valid data
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=1.5)
            Z = gaussian_filter(Z, sigma=0.75)
            Z = gaussian_filter(Z, sigma=0.5)
        
        return X, Y, Z
        
    except Exception as e:
        st.error(f"Error creating IV surface: {str(e)}")
        return None, None, None

#Streamlit UI
st.title("Ez Options")

# Modify the reset_session_state function to preserve color settings
def reset_session_state():
    """Reset all session state variables except for essential ones"""
    # Keep track of keys we want to preserve
    preserved_keys = {
        'current_page', 
        'previous_page',
        'loading_complete',
        'initialized', 
        'saved_ticker', 
        'call_color', 
        'put_color',
        'vix_color',
        'show_calls', 
        'show_puts',
        'show_net',
        'strike_range',
        'chart_type',
        'chart_text_size',
        'refresh_rate',
        'intraday_chart_type',
        'candlestick_type',
        'show_vix_overlay',
        'gex_type',
        'abs_gex_opacity',
        'intraday_exposure_levels',
        'show_straddle',
        'show_technical_indicators',
        'selected_indicators',
        'ema_periods',
        'sma_periods',
        'bollinger_period',
        'bollinger_std',
        'rsi_period',
        'fibonacci_levels',
        'vwap_enabled',
        'exposure_metric',
        'exposure_perspective',
        'delta_adjusted_exposures',
        'calculate_in_notional',
        'global_selected_expiries',
        'saved_exposure_heatmap_type',
        'intraday_level_count',
        'show_sd_move'
    }
    
    # Initialize visibility settings if they don't exist
    if 'show_calls' not in st.session_state:
        st.session_state.show_calls = True
    if 'show_puts' not in st.session_state:
        st.session_state.show_puts = True
    if 'show_net' not in st.session_state:
        st.session_state.show_net = True
    
    preserved_values = {key: st.session_state[key] 
                       for key in preserved_keys 
                       if key in st.session_state}
    
    # Clear everything safely
    for key in list(st.session_state.keys()):
        if key not in preserved_keys:
            try:
                del st.session_state[key]
            except KeyError:
                pass
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value

    # Reset expiry selection keys explicitly
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'color_expiry_multi',
        'max_pain_expiry_multi',
        'exposure_heatmap_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add near the top with other session state initializations
if 'global_selected_expiries' not in st.session_state:
    st.session_state.global_selected_expiries = []

@st.fragment
def expiry_selector_fragment(page_name, available_dates):
    """Fragment for expiry date selection that properly resets"""
    container = st.empty()
    
    # Initialize global state if needed
    if 'global_selected_expiries' not in st.session_state:
        st.session_state.global_selected_expiries = []
    
    widget_key = f"{page_name}_expiry_selector"
    
    # Initialize previous selection state if not exists
    if f"{widget_key}_prev" not in st.session_state:
        st.session_state[f"{widget_key}_prev"] = []
    
    with container:
        # For implied probabilities page, use single select
        if page_name == "Implied Probabilities":
            # Single select - default to nearest (first available)
            # Do not use global state for default
            # Do not update global state
            
            selected_date = st.selectbox(
                "Select Expiration Date:",
                options=available_dates,
                index=0,
                key=widget_key
            )
            selected = [selected_date] if selected_date else []
            
        else:
            # Multi select - use global state
            # Filter global selection to only include available dates
            default_selection = [d for d in st.session_state.global_selected_expiries if d in available_dates]
            
            # Default to nearest expiry if nothing is selected
            if not default_selection:
                nearest = get_nearest_expiry(available_dates)
                if nearest:
                    default_selection = [nearest]

            selected = st.multiselect(
                "Select Expiration Date(s):",
                options=available_dates,
                default=default_selection,
                key=widget_key
            )
            
            # Check if selection changed
            if selected != st.session_state.get(f"{widget_key}_prev"):
                st.session_state.global_selected_expiries = selected
                st.session_state[f"{widget_key}_prev"] = selected.copy()
                if selected:  # Only rerun if there are selections
                    st.rerun()
    
    return selected, container

def handle_page_change(new_page):
    """Handle page navigation and state management"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = new_page
        return True
    
    if st.session_state.current_page != new_page:
        # Clear page-specific widget state to force default reload
        old_widget_key = f"{st.session_state.current_page}_expiry_selector"
        if old_widget_key in st.session_state:
            del st.session_state[old_widget_key]
            
        if 'expiry_selector_container' in st.session_state:
            st.session_state.expiry_selector_container.empty()
        
        # Clear previous page state
        clear_page_state()
        
        st.session_state.current_page = new_page
        reset_session_state()
        st.rerun()
        return True
    
    return False

# Save selected ticker
def save_ticker(ticker):
    st.session_state.saved_ticker = ticker

# Market Maker Functions
def get_latest_business_day():
    """Get the latest business day (Monday-Friday) from 24 hours ago."""
    now = get_now_et()
    twenty_four_hours_ago = now - timedelta(hours=24)
    
    # Start with the date from 24 hours ago
    target_date = twenty_four_hours_ago
    
    # If the date from 24 hours ago is a weekend, find the most recent weekday
    while target_date.weekday() > 4:  # Saturday=5, Sunday=6
        target_date -= timedelta(days=1)
    
    # Always return a business day, even if it's more than 24 hours ago
    # This ensures we get market maker data for the most recent trading day
    return target_date

def should_run_script():
    """Check if script should run (based on whether there's a trading day within 24 hours)."""
    # The script should run if there's a valid trading day within the last 24 hours
    # This will be determined by get_latest_business_day(), so we'll let it run
    # and let that function determine if there's valid data to fetch
    return True

def get_params_for_date(target_date, symbol=None, symbol_type="U"):
    """Get API parameters for a specific date and optional symbol."""
    # Safety check: ensure we never request future dates
    today = get_now_et().date()
    if target_date.date() > today:
        target_date = get_now_et() - timedelta(days=1)  # Use yesterday instead
        while target_date.weekday() > 4:  # Skip weekends
            target_date -= timedelta(days=1)
    
    params = {
        "format": "csv",
        "volumeQueryType": "O",  # Options
        "symbolType": "ALL" if symbol is None else symbol_type,  # ALL for all symbols, O/U for specific
        "reportType": "D",       # Daily report (can be modified to W or M)
        "accountType": "M",      # Market Maker
        "productKind": "ALL",    # All product types (Equity, Index, etc.)
        "porc": "BOTH",          # Calls and Puts
        "reportDate": target_date.strftime("%Y%m%d")  # Date in YYYYMMDD format
    }
    
    # If a specific symbol is provided, add it as a filter parameter
    if symbol:
        params["symbol"] = symbol.upper()
    
    return params

def process_market_maker_data(csv_data):
    """Process market maker CSV data and return summary statistics"""
    try:
        from io import StringIO
        
        # Skip any header lines that are not part of the CSV data
        lines = csv_data.strip().split('\n')
        
        # Find the first line that looks like CSV data (has commas and starts with a number)
        start_index = 0
        for i, line in enumerate(lines):
            if ',' in line and line.strip() and line.strip()[0].isdigit():
                start_index = i
                break
        
        # Reconstruct CSV data from the actual data lines
        csv_lines = lines[start_index:]
        clean_csv = '\n'.join(csv_lines)
        
        # Read CSV with proper column names based on OCC layout
        df = pd.read_csv(StringIO(clean_csv), header=None, names=[
            'Quantity',           # Column 1: Volume
            'Underlying_Symbol',  # Column 2: Underlying symbol  
            'Options_Symbol',     # Column 3: Options symbol
            'Account_Type',       # Column 4: Account type (C/F/M)
            'Call_Put_Indicator', # Column 5: Call/Put indicator (C/P)
            'Exchange',           # Column 6: Exchange
            'Activity_Date',      # Column 7: Activity date
            'Series_Date'         # Column 8: Series/contract date
        ])
        
        if df.empty:
            return None
        
        # Initialize summary data
        summary = {
            'total_volume': 0,
            'call_volume': 0,
            'put_volume': 0,
            'call_percentage': 0,
            'put_percentage': 0,
            'raw_data': df
        }
        
        # Filter for Market Maker data only
        mm_data = df[df['Account_Type'] == 'M']
        
        if mm_data.empty:
            # If no MM data, use all data
            mm_data = df
        
        # Calculate call and put volumes using the Call_Put_Indicator column
        call_data = mm_data[mm_data['Call_Put_Indicator'] == 'C']
        put_data = mm_data[mm_data['Call_Put_Indicator'] == 'P']
        
        summary['call_volume'] = call_data['Quantity'].sum() if not call_data.empty else 0
        summary['put_volume'] = put_data['Quantity'].sum() if not put_data.empty else 0
        summary['total_volume'] = summary['call_volume'] + summary['put_volume']
        
        if summary['total_volume'] > 0:
            summary['call_percentage'] = (summary['call_volume'] / summary['total_volume']) * 100
            summary['put_percentage'] = (summary['put_volume'] / summary['total_volume']) * 100
        

        
        return summary
        
    except Exception as e:
        # Fallback to original logic if the structured approach fails
        try:
            df = pd.read_csv(StringIO(csv_data))
            if df.empty:
                return None
                
            summary = {
                'total_volume': 0,
                'call_volume': 0,
                'put_volume': 0,
                'call_percentage': 0,
                'put_percentage': 0,
                'raw_data': df
            }
            
            # Try to find volume column
            volume_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['quantity', 'volume', 'vol']):
                    volume_col = col
                    break
            
            if volume_col is None and len(df.select_dtypes(include=[np.number]).columns) > 0:
                volume_col = df.select_dtypes(include=[np.number]).columns[0]
            
            if volume_col:
                # Try to find call/put indicator column
                cp_col = None
                for col in df.columns:
                    if df[col].dtype == 'object':
                        unique_vals = df[col].unique()
                        if any('C' in str(unique_vals)) and any('P' in str(unique_vals)):
                            cp_col = col
                            break
                
                if cp_col:
                    call_data = df[df[cp_col] == 'C']
                    put_data = df[df[cp_col] == 'P']
                    
                    summary['call_volume'] = call_data[volume_col].sum() if not call_data.empty else 0
                    summary['put_volume'] = put_data[volume_col].sum() if not put_data.empty else 0
                    summary['total_volume'] = summary['call_volume'] + summary['put_volume']
                    
                    if summary['total_volume'] > 0:
                        summary['call_percentage'] = (summary['call_volume'] / summary['total_volume']) * 100
                        summary['put_percentage'] = (summary['put_volume'] / summary['total_volume']) * 100
            
            return summary
            
        except Exception:
            return None

def create_market_maker_charts(summary_data):
    """Create charts for market maker data visualization"""
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    # Ensure we have valid data for charts
    call_vol = max(summary_data['call_volume'], 0)
    put_vol = max(summary_data['put_volume'], 0)
    
    # If both volumes are zero, create placeholder charts
    if call_vol == 0 and put_vol == 0:
        call_vol, put_vol = 1, 1  # Equal placeholder values
    
    # Call/Put Volume Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Calls', 'Puts'],
        values=[call_vol, put_vol],
        marker_colors=[call_color, put_color],
        hole=0.4,
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig_pie.update_layout(
        title=dict(
            text="Market Maker Call/Put Volume Distribution",
            x=0.5,
            xanchor='center',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        template="plotly_dark",
        height=500,  # Increased height for bigger chart
        showlegend=False,  # Remove legend
        margin=dict(t=80, b=80, l=80, r=80)  # Add margins for better spacing
    )
    
    # Volume Bar Chart
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=['Calls', 'Puts'],
        y=[call_vol, put_vol],
        marker_color=[call_color, put_color],
        text=[f"{summary_data['call_volume']:,}", f"{summary_data['put_volume']:,}"],
        textposition='auto',
        name='Volume'
    ))
    
    fig_bar.update_layout(
        title=dict(
            text="Market Maker Volume Breakdown",
            x=0.5,
            xanchor='center',
            font=dict(size=st.session_state.chart_text_size + 3)
        ),
        xaxis_title=dict(
            text="Option Type",
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text="Volume",
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        height=500,  # Increased height for bigger chart
        showlegend=False,
        margin=dict(t=80, b=80, l=80, r=80)  # Add margins for better spacing
    )
    
    return fig_pie, fig_bar

def download_volume_csv(symbol=None, symbol_type="U", expiry_date=None):
    """Download market maker volume data from OCC API"""
    BASE_URL = "https://marketdata.theocc.com/volume-query"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Check if we should run the script
    if not should_run_script():
        return None, "Script is currently disabled."
    
    # Clean symbol for OCC API (remove ^ prefix from index symbols)
    clean_symbol = symbol
    if symbol and symbol.startswith('^'):
        clean_symbol = symbol[1:]  # Remove the ^ prefix
    
    # The API report date should be from the latest business day from 24 hours ago
    api_report_date = get_latest_business_day()
    
    # If expiry_date is provided, use it for display purposes
    # The actual API still uses the api_report_date for the reportDate parameter
    display_date = expiry_date if expiry_date else api_report_date
    
    params = get_params_for_date(api_report_date, clean_symbol, symbol_type)
    
    try:
        # Make the GET request to the OCC volume query endpoint
        response = requests.get(BASE_URL, params=params, headers=HEADERS)
        
        # Check if the request was successful
        if response.status_code == 200:
            if expiry_date:
                # Convert expiry_date string to datetime for formatting
                try:
                    if isinstance(expiry_date, str):
                        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                        display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_dt.strftime('%Y-%m-%d (%A)')} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
                    else:
                        display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date.strftime('%Y-%m-%d (%A)')} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
                except:
                    display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
            else:
                display_text = f"Market Maker Positioning Data Retrieved Successfully for {api_report_date.strftime('%Y-%m-%d (%A)')}"
            return response.text, display_text
        else:
            # Try previous business days (up to 5 days back)
            for days_back in range(1, 6):  # Try up to 5 days back
                fallback_date = get_now_et() - timedelta(days=days_back)
                
                # Skip weekends
                if fallback_date.weekday() > 4:
                    continue
                
                params_fallback = get_params_for_date(fallback_date, clean_symbol, symbol_type)
                
                try:
                    response_fallback = requests.get(BASE_URL, params=params_fallback, headers=HEADERS)
                    if response_fallback.status_code == 200:
                        if expiry_date:
                            try:
                                if isinstance(expiry_date, str):
                                    expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                                    fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_dt.strftime('%Y-%m-%d (%A)')} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                                else:
                                    fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date.strftime('%Y-%m-%d (%A)')} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                            except:
                                fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                        else:
                            fallback_text = f"Market Maker Positioning Data Retrieved Successfully for {fallback_date.strftime('%Y-%m-%d (%A)')}"
                        return response_fallback.text, fallback_text
                except requests.exceptions.RequestException:
                    continue
            
            return None, f"Error: Failed to download CSV. Status code: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Error during request: {e}"

st.sidebar.title("📊 Navigation")

# Define pages with their corresponding icons
page_icons = {
    "Dashboard": "🏠",
    "OI & Volume": "📈", 
    "Gamma Exposure": "🔺",
    "Delta Exposure": "📊",
    "Vanna Exposure": "🌊",
    "Charm Exposure": "⚡",
    "Speed Exposure": "🚀",
    "Vomma Exposure": "💫",
    "Color Exposure": "🌈",
    "Delta-Adjusted Value Index": "📉",
    "Max Pain": "🎯",
    "Exposure Heatmap": "🔥",
    "GEX Surface": "🗻",
    "IV Surface": "🌐",
    "Implied Probabilities": "🎲",
    "Analysis": "🔍"
}

pages = ["Dashboard", "OI & Volume", "Gamma Exposure", "Delta Exposure", 
          "Vanna Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Color Exposure", "Delta-Adjusted Value Index", "Max Pain", "Exposure Heatmap", "GEX Surface", "IV Surface",
          "Implied Probabilities", "Analysis"]

# Create page options with icons
page_options = [f"{page_icons[page]} {page}" for page in pages]

# Track the previous page in session state
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None

selected_page_with_icon = st.sidebar.radio("Select a page:", page_options)

# Extract the actual page name (remove icon and space)
new_page = selected_page_with_icon.split(" ", 1)[1]

# Check if the page has changed
if st.session_state.previous_page != new_page:
    st.session_state.previous_page = new_page
    handle_page_change(new_page)
    # Clear out any page-specific expiry selections
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'color_expiry_multi',
        'exposure_heatmap_expiry_multi',
        'implied_probabilities_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add after st.sidebar.title("Navigation")
def chart_settings():
    with st.sidebar.expander("Chart Settings", expanded=False):
        # Greek Exposure Settings - FIRST SETTING
        st.write("Greek Exposure Settings:")
        
        # Initialize exposure metric setting
        if 'exposure_metric' not in st.session_state:
            # Migration from old setting if exists
            if st.session_state.get('use_volume_for_greeks', False):
                st.session_state.exposure_metric = 'Volume'
            else:
                st.session_state.exposure_metric = 'Open Interest'
        
        st.selectbox(
            "Exposure Calculation Metric:",
            options=['Open Interest', 'Volume', 'Volume Weighted by OI'],
            index=['Open Interest', 'Volume', 'Volume Weighted by OI'].index(st.session_state.exposure_metric),
            key='exposure_metric',
            help="Choose the metric used for weighting Greek exposures"
        )

        # Initialize perspective setting
        if 'exposure_perspective' not in st.session_state:
            st.session_state.exposure_perspective = "Customer"
        
        st.selectbox(
            "Exposure Perspective:",
            options=["Customer", "Dealer"],
            index=["Customer", "Dealer"].index(st.session_state.exposure_perspective),
            key='exposure_perspective',
            help="Customer: View calculated exposures for Long option positions.\nDealer: View exposures for Short option positions (Inverse of Customer)."
        )

        # Initialize delta-adjusted exposures setting
        if 'delta_adjusted_exposures' not in st.session_state:
            st.session_state.delta_adjusted_exposures = False  # Default to not delta-adjusted
        
        st.checkbox(
            "Delta-Adjusted Exposures",
            value=st.session_state.delta_adjusted_exposures,
            key='delta_adjusted_exposures',
            help="When enabled, all Greek exposures (Gamma, Vanna, Charm, Speed, Vomma) will be multiplied by Delta to show delta-adjusted values"
        )

        # Initialize notional calculation setting
        if 'calculate_in_notional' not in st.session_state:
            st.session_state.calculate_in_notional = True

        st.checkbox(
            "Calculate Exposures in Notional ($)",
            value=st.session_state.calculate_in_notional,
            key='calculate_in_notional',
            help="If enabled, exposures are calculated in Dollar value (multiplying by Spot Price). If disabled, they are calculated in Underlying Units (Shares for stocks, Index Units for indices)."
        )

        st.write("Colors:")
        st.color_picker("Calls", st.session_state.call_color, key='call_color')
        st.color_picker("Puts", st.session_state.put_color, key='put_color')
        
        # Add intraday chart type selection
        if 'intraday_chart_type' not in st.session_state:
            st.session_state.intraday_chart_type = 'Candlestick'
        
        if 'candlestick_type' not in st.session_state:
            st.session_state.candlestick_type = 'Filled'
        
        st.selectbox(
            "Intraday Chart Type:",
            options=['Candlestick', 'Line'],
            index=['Candlestick', 'Line'].index(st.session_state.intraday_chart_type),
            key='intraday_chart_type'
        )
        
        # Only show candlestick type selection when candlestick chart is selected
        if st.session_state.intraday_chart_type == 'Candlestick':
            st.selectbox(
                "Candlestick Style:",
                options=['Filled', 'Hollow', 'Heikin Ashi'],
                index=['Filled', 'Hollow', 'Heikin Ashi'].index(st.session_state.candlestick_type),
                key='candlestick_type'
            )

        if 'show_vix_overlay' not in st.session_state:
            st.session_state.show_vix_overlay = False
        
        # Group VIX settings together
        st.write("VIXY Settings:")
        st.checkbox("VIXY Overlay", value=st.session_state.show_vix_overlay, key='show_vix_overlay')
        
        if st.session_state.show_vix_overlay:
            st.color_picker("VIXY Color", st.session_state.vix_color, key='vix_color')

        # Technical Indicators Settings
        st.write("Technical Indicators:")
        
        # Initialize technical indicators settings
        if 'show_technical_indicators' not in st.session_state:
            st.session_state.show_technical_indicators = False
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = []
        if 'ema_periods' not in st.session_state:
            st.session_state.ema_periods = [9, 21, 50]
        if 'sma_periods' not in st.session_state:
            st.session_state.sma_periods = [20, 50]
        if 'bollinger_period' not in st.session_state:
            st.session_state.bollinger_period = 20
        if 'bollinger_std' not in st.session_state:
            st.session_state.bollinger_std = 2.0
        if 'rsi_period' not in st.session_state:
            st.session_state.rsi_period = 14
        if 'fibonacci_levels' not in st.session_state:
            st.session_state.fibonacci_levels = True
        if 'vwap_enabled' not in st.session_state:
            st.session_state.vwap_enabled = False
        
        show_technical = st.checkbox("Enable Technical Indicators", value=st.session_state.show_technical_indicators, key="show_technical_indicators")
        
        # Clear selected indicators when disabling
        if not show_technical and st.session_state.selected_indicators:
            st.session_state.selected_indicators = []
        
        if show_technical:
            # Available indicators
            available_indicators = [
                "EMA (Exponential Moving Average)",
                "SMA (Simple Moving Average)", 
                "Bollinger Bands",
                "RSI (Relative Strength Index)",
                "VWAP (Volume Weighted Average Price)",
                "Fibonacci Retracements"
            ]
            
            st.multiselect(
                "Select Indicators:",
                available_indicators,
                default=st.session_state.selected_indicators,
                key="selected_indicators"
            )
            
            # EMA Settings
            if "EMA (Exponential Moving Average)" in st.session_state.selected_indicators:
                st.write("**EMA Settings:**")
                ema_input = st.text_input(
                    "EMA Periods (comma-separated)",
                    value=",".join(map(str, st.session_state.ema_periods)),
                    help="e.g., 9,21,50",
                    key="ema_periods_input"
                )
                try:
                    ema_periods = [int(x.strip()) for x in ema_input.split(",") if x.strip()]
                    if ema_periods != st.session_state.ema_periods:
                        st.session_state.ema_periods = ema_periods
                except:
                    st.warning("Invalid EMA periods format. Use comma-separated integers.")
            
            # SMA Settings  
            if "SMA (Simple Moving Average)" in st.session_state.selected_indicators:
                st.write("**SMA Settings:**")
                sma_input = st.text_input(
                    "SMA Periods (comma-separated)",
                    value=",".join(map(str, st.session_state.sma_periods)),
                    help="e.g., 20,50",
                    key="sma_periods_input"
                )
                try:
                    sma_periods = [int(x.strip()) for x in sma_input.split(",") if x.strip()]
                    if sma_periods != st.session_state.sma_periods:
                        st.session_state.sma_periods = sma_periods
                except:
                    st.warning("Invalid SMA periods format. Use comma-separated integers.")
            
            # Bollinger Bands Settings
            if "Bollinger Bands" in st.session_state.selected_indicators:
                st.write("**Bollinger Bands Settings:**")
                st.number_input("Period", min_value=5, max_value=50, value=st.session_state.bollinger_period, key="bollinger_period")
                st.number_input("Standard Deviations", min_value=1.0, max_value=3.0, value=st.session_state.bollinger_std, step=0.1, key="bollinger_std")
            
            # RSI Settings
            if "RSI (Relative Strength Index)" in st.session_state.selected_indicators:
                st.write("**RSI Settings:**")
                st.number_input("RSI Period", min_value=5, max_value=30, value=st.session_state.rsi_period, key="rsi_period")
            
        if 'chart_text_size' not in st.session_state:
            st.session_state.chart_text_size = 12  # Default text size
            
        st.number_input(
            "Chart Text Size",
            min_value=10,
            max_value=30,
            value=st.session_state.chart_text_size,
            step=1,
            help="Adjust the size of text in charts (titles, labels, etc.)",
            key='chart_text_size'
        )
        
        st.write("Show/Hide Elements:")
        # Initialize visibility settings if not already set
        if 'show_calls' not in st.session_state:
            st.session_state.show_calls = False
        if 'show_puts' not in st.session_state:
            st.session_state.show_puts = False
        if 'show_net' not in st.session_state:
            st.session_state.show_net = True

        # Visibility toggles
        st.checkbox("Show Calls", value=st.session_state.show_calls, key='show_calls')
        st.checkbox("Show Puts", value=st.session_state.show_puts, key='show_puts')
        st.checkbox("Show Net", value=st.session_state.show_net, key='show_net')

        # Initialize strike range in session state (as percentage)
        if 'strike_range' not in st.session_state:
            st.session_state.strike_range = 2.0  # Default 2%
        
        # Add strike range control (as percentage)
        st.number_input(
            "Strike Range (% from current price)",
            min_value=0.1,
            max_value=50.0,
            value=st.session_state.strike_range,
            step=0.1,
            format="%.1f",
            help="Percentage range from current price (e.g., 1.0 = ±1%)",
            key="strike_range"
        )

        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = 'Bar'  # Default chart type

        st.selectbox(
            "Chart Type:",
            options=['Bar', 'Horizontal Bar', 'Scatter', 'Line', 'Area'],
            index=['Bar', 'Horizontal Bar', 'Scatter', 'Line', 'Area'].index(st.session_state.chart_type),
            key='chart_type'
        )

         
        if 'gex_type' not in st.session_state:
            st.session_state.gex_type = 'Net'  # Default to Net GEX
        
        st.selectbox(
            "Gamma Exposure Type:",
            options=['Net', 'Absolute'],
            index=['Net', 'Absolute'].index(st.session_state.gex_type),
            key='gex_type'
        )

        # Add opacity setting for Absolute Gamma Background
        if 'abs_gex_opacity' not in st.session_state:
            st.session_state.abs_gex_opacity = 0.0  # Default 0 (hidden)
            
        st.slider(
            "Absolute Gamma Background Opacity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.abs_gex_opacity,
            step=0.05,
            key='abs_gex_opacity',
            help="Opacity of the absolute gamma exposure area chart in the background"
        )

        # Add intraday chart level settings
        st.write("Intraday Chart Levels:")
        
        # Initialize exposure levels setting
        if 'intraday_exposure_levels' not in st.session_state:
            # Migrate from old settings
            defaults = []
            if st.session_state.get('show_gex_levels', True):
                defaults.append('GEX')
            if st.session_state.get('show_dex_levels', False):
                defaults.append('DEX')
            st.session_state.intraday_exposure_levels = defaults

        if 'intraday_level_count' not in st.session_state:
            st.session_state.intraday_level_count = 5

        if 'show_straddle' not in st.session_state:
            st.session_state.show_straddle = False  # Default to not showing Straddle
            
        if 'show_sd_move' not in st.session_state:
            st.session_state.show_sd_move = False  # Default to not showing 1 SD Move

        # Exposure levels multiselect
        exposure_options = ['GEX', 'DEX', 'VEX', 'Charm', 'Speed', 'Vomma', 'Color']
        st.multiselect(
            "Show Levels for:",
            options=exposure_options,
            default=st.session_state.intraday_exposure_levels,
            key='intraday_exposure_levels'
        )
        
        st.number_input(
            "Number of Levels to Show",
            min_value=1,
            max_value=50,
            value=st.session_state.intraday_level_count,
            step=1,
            key='intraday_level_count'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox(
                "Show Straddle", 
                value=st.session_state.show_straddle, 
                key='show_straddle',
                help="Plots top and bottom breakeven lines for the ATM Straddle. Historically, price stays within this range ~55% of the time."
            )
        with col2:
            st.checkbox(
                "Show 1 SD Move", 
                value=st.session_state.show_sd_move, 
                key='show_sd_move',
                help="Plots the expected 1 Standard Deviation move. Statistically, price stays within this range ~68.2% of the time."
            )

        # Add refresh rate control before chart type
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = 10  # Default refresh rate
        
        def on_refresh_rate_change():
            st.cache_data.clear()
            
        st.number_input(
            "Auto-Refresh Rate (seconds)",
            min_value=10,
            max_value=300,
            value=int(st.session_state.refresh_rate),
            step=1,
            help="How often to auto-refresh the page (minimum 10 seconds)",
            key='refresh_rate',
            on_change=on_refresh_rate_change
        )

# Call the regular function instead of the fragment
chart_settings()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <a href="https://github.com/EazyDuz1t/EzOptions" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub Repo" style="margin-bottom: 10px;">
        </a>
        <br>
        <a href="https://discord.com/users/eazy101" target="_blank">
            <img src="https://img.shields.io/badge/Discord-eazy101-7289DA?logo=discord&logoColor=white" alt="Discord User">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Use the saved ticker and expiry date if available
saved_ticker = st.session_state.get("saved_ticker", "")
saved_expiry_date = st.session_state.get("saved_expiry_date", None)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = get_now_et().date()
        # For future dates, ensure they're treated as valid
        return expiry_date >= current_market_date
    except Exception:
        return False

def compute_greeks_and_charts(ticker, expiry_date_str, page_key, S):
    """Compute greeks and create charts for options data"""
    if not expiry_date_str:
        st.warning("Please select an expiration date.")
        return None, None, None, None, None, None
        
    calls, puts = fetch_options_for_date(ticker, expiry_date_str, S)
    if calls.empty and puts.empty:
        st.warning("No options data available for this ticker.")
        return None, None, None, None, None, None

    combined = pd.concat([calls, puts])
    combined = combined.dropna(subset=['extracted_expiry'])
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    calls = calls[calls['extracted_expiry'] == selected_expiry].copy()
    puts = puts[puts['extracted_expiry'] == selected_expiry].copy()

    # Always use get_current_price to ensure consistent price source
    if S is None:
        st.error("Could not fetch underlying price.")
        return None, None, None, None, None, None

    S = float(S)  # Ensure price is float
    
    # Fetch dividend yield
    try:
        if ticker == "MARKET":
            q = 0
        else:
            stock_info = get_ticker_object(ticker).info
            q = stock_info.get('dividendYield', 0)
            if q is None: q = 0
    except:
        q = 0
    
    # Calculate time to expiration more precisely
    today = get_now_et().date()
    t = calculate_time_to_expiration(selected_expiry)
    
    # Check if expired
    if t <= 0:
         if selected_expiry < today:
             st.error("The selected expiration date is in the past!")
             return None, None, None, None, None, None
         t = 1e-5 # Minimum time for 0DTE at close

    t = max(t, 1e-5) # Ensure positive and non-zero
    r = st.session_state.get('risk_free_rate', 0.04)

    def get_valid_sigma(row, flag):
        # 1. Try Manual Calculation (Preferred: uses Mid Price)
        try:
            # Use Mid Price
            bid = row.get('bid', 0)
            ask = row.get('ask', 0)
            price = (bid + ask) / 2
                
            if price > 0:
                calculated_sigma = calculate_implied_volatility(price, S, row['strike'], t, r, flag, q)
                if calculated_sigma is not None and 0 < calculated_sigma <= 5.0:
                    return calculated_sigma
        except Exception:
            pass
            
        # 2. Fallback to YFinance provided IV
        try:
            yf_iv = row.get('impliedVolatility')
            if yf_iv is not None and 0 < yf_iv <= 5.0:
                return yf_iv
        except:
            pass
            
        return None

    # Compute Greeks for Gamma, Vanna, Delta, Charm, Speed, Vomma and Color
    def compute_all_greeks(row, flag):
        sigma = get_valid_sigma(row, flag)
        if sigma is None:
            return pd.Series([None] * 7)
        try:
            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma, r, q)
            charm_val = calculate_charm(flag, S, row["strike"], t, sigma, r, q)
            speed_val = calculate_speed(flag, S, row["strike"], t, sigma, r, q)
            vomma_val = calculate_vomma(flag, S, row["strike"], t, sigma, r, q)
            color_val = calculate_color(flag, S, row["strike"], t, sigma, r, q)
            
            return pd.Series([gamma_val, vanna_val, delta_val, charm_val, speed_val, vomma_val, color_val])
        except Exception:
            return pd.Series([None] * 7)

    greek_columns = ["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma", "calc_color"]

    if not calls.empty:
        calls[greek_columns] = calls.apply(lambda row: compute_all_greeks(row, "c"), axis=1)
    
    if not puts.empty:
        puts[greek_columns] = puts.apply(lambda row: compute_all_greeks(row, "p"), axis=1)

    calls = calls.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma", "calc_color"])
    puts = puts.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma", "calc_color"])

    # Determine which metric to use based on settings
    metric_type = st.session_state.get('exposure_metric', 'Open Interest')
    
    if metric_type == 'Volume':
        calls_metric = calls['volume']
        puts_metric = puts['volume']
    elif metric_type == 'Volume Weighted by OI':
        # Average of Volume and OI: (Volume + OI) / 2
        calls_metric = (calls['volume'].fillna(0) + calls['openInterest'].fillna(0)) / 2
        puts_metric = (puts['volume'].fillna(0) + puts['openInterest'].fillna(0)) / 2
    else: # Open Interest
        calls_metric = calls['openInterest']
        puts_metric = puts['openInterest']

    # Determine if we should calculate in notional (dollars) or underlying units (shares/index units)
    use_notional = st.session_state.get('calculate_in_notional', True)
    spot_multiplier = S if use_notional else 1.0

    # GEX = Gamma * Metric * Contract Size * Spot Price^2 * 0.01 (Dollar Gamma per 1% move in underlying)
    calls["GEX"] = calls["calc_gamma"] * calls_metric * 100 * S * spot_multiplier * 0.01
    puts["GEX"] = puts["calc_gamma"] * puts_metric * 100 * S * spot_multiplier * 0.01
    
    # VEX = Vanna * Metric * Contract Size * Spot Price * 0.01 (Dollar Vanna per 1 vol point change)
    calls["VEX"] = calls["calc_vanna"] * calls_metric * 100 * spot_multiplier * 0.01
    puts["VEX"] = puts["calc_vanna"] * puts_metric * 100 * spot_multiplier * 0.01
    
    # DEX = Delta * Metric * Contract Size * Spot Price (Dollar Delta Exposure)
    calls["DEX"] = calls["calc_delta"] * calls_metric * 100 * spot_multiplier
    puts["DEX"] = puts["calc_delta"] * puts_metric * 100 * spot_multiplier
    
    # Charm = Charm * Metric * Contract Size * Spot Price / 365 (Dollar Charm per 1 day decay)
    calls["Charm"] = calls["calc_charm"] * calls_metric * 100 * spot_multiplier / 365.0
    puts["Charm"] = puts["calc_charm"] * puts_metric * 100 * spot_multiplier / 365.0
    
    # Speed = Speed * Metric * Contract Size * Spot Price^2 * 0.01 (Dollar Speed per 1% move)
    calls["Speed"] = calls["calc_speed"] * calls_metric * 100 * S * spot_multiplier * 0.01
    puts["Speed"] = puts["calc_speed"] * puts_metric * 100 * S * spot_multiplier * 0.01
    
    # Vomma = Vomma * Metric * Contract Size * 0.01 (Dollar Vomma per 1 vol point change)
    calls["Vomma"] = calls["calc_vomma"] * calls_metric * 100 * 0.01
    puts["Vomma"] = puts["calc_vomma"] * puts_metric * 100 * 0.01

    # Color = Color * Metric * Contract Size * Spot Price^2 * 0.01 / 365 (Dollar Color per 1% move per 1 day decay)
    # Color is dGamma/dt. GEX is Dollar Gamma. So Color Exposure is d(GEX)/dt.
    # GEX = Gamma * Metric * 100 * S * Spot * 0.01.
    # So d(GEX)/dt = Color * Metric * 100 * S * Spot * 0.01 / 365.
    calls["Color"] = calls["calc_color"] * calls_metric * 100 * S * spot_multiplier * 0.01 / 365.0
    puts["Color"] = puts["calc_color"] * puts_metric * 100 * S * spot_multiplier * 0.01 / 365.0

    # Apply delta adjustment if enabled
    if st.session_state.get('delta_adjusted_exposures', False):
        # Multiply all exposures (except DEX which is already delta-based) by delta (probability)
        # Use absolute delta to weight by probability without flipping signs for puts
        calls["GEX"] = calls["GEX"] * calls["calc_delta"].abs()
        puts["GEX"] = puts["GEX"] * puts["calc_delta"].abs()
        
        calls["VEX"] = calls["VEX"] * calls["calc_delta"].abs()
        puts["VEX"] = puts["VEX"] * puts["calc_delta"].abs()
        
        calls["Charm"] = calls["Charm"] * calls["calc_delta"].abs()
        puts["Charm"] = puts["Charm"] * puts["calc_delta"].abs()
        
        calls["Speed"] = calls["Speed"] * calls["calc_delta"].abs()
        puts["Speed"] = puts["Speed"] * puts["calc_delta"].abs()
        
        calls["Vomma"] = calls["Vomma"] * calls["calc_delta"].abs()
        puts["Vomma"] = puts["Vomma"] * puts["calc_delta"].abs()

        calls["Color"] = calls["Color"] * calls["calc_delta"].abs()
        puts["Color"] = puts["Color"] * puts["calc_delta"].abs()

    return calls, puts, S, t, selected_expiry, today

def get_exposure_explanation(exposure_type):
    """Get explanation text for each exposure type based on selected perspective."""
    perspective = st.session_state.get('exposure_perspective', 'Customer')
    is_dealer = perspective == 'Dealer'
    
    # Precise Explanations of Market Mechanics
    explanations = {
        "GEX": {
            "title": "Gamma Exposure (GEX)",
            "description": "Gamma measures the rate of change of Delta. It indicates how much hedging activity is required as price moves.",
            "customer": """
**Customer Perspective (Long Options):**
- **Calls (Positive GEX):** Customer is Long Call / Dealer is Short Call (Negative Gamma). As price rises, Dealer must **BUY** to hedge.
- **Puts (Positive GEX):** Customer is Long Put / Dealer is Short Put (Negative Gamma). As price falls, Dealer must **SELL** to hedge.
- **Implication:** Positive Customer GEX = Dealer Short Gamma = **Volatility Amplification** (Dealer chases price).
- **Net Negative GEX:** Customer is Short / Dealer is Long. Dealer hedges by buying low/selling high (**Volatility Suppression**).
            """,
            "dealer": """
**Dealer Perspective (Short Options):**
- **Negative GEX:** Dealers are Short Gamma. They must **Buy High** and **Sell Low** to stay hedged.
- **Market Impact:** This hedging activity **amplifies** price movement (High Volatility).
- **Positive GEX:** Dealers are Long Gamma. They **Buy Low** and **Sell High**.
- **Market Impact:** This hedging activity **suppresses** price movement (Low Volatility / Pinning).
            """
        },
        "VEX": {
            "title": "Vanna Exposure (VEX)",
            "description": "Vanna measures delta sensitivity to changes in Implied Volatility (dDelta/dVol).",
            "customer": """
**Customer Perspective (Long Options):**
- **Call Vanna (Positive):** Customer Long OTM Call. If IV rises, Call Delta increases. Dealer (Short) becomes more Short Delta -> Must **BUY**.
- **Put Vanna (Negative):** Customer Long OTM Put. If IV rises, Put Delta becomes more Negative. Dealer (Short) becomes more Long Delta (Short Put Delta increases). Dealer must **SELL** to hedge (short more stock).
- **Flow:** Rising IV forces Dealers to Buy against Calls and Sell against Puts.
            """,
            "dealer": """
**Dealer Perspective (Short Options):**
- **Negative Vanna (Short Calls):** IV Rise -> Dealer Short Delta becomes more negative -> Dealer **Buys**.
- **Positive Vanna (Short Puts):** IV Rise -> Dealer Short Delta becomes more positive -> Dealer **Sells**.
- **Key Insight:** Vanna flows often drive the market during VIX spikes or crushes.
            """
        },
        "DEX": {
            "title": "Delta Exposure (DEX)",
            "description": "Delta Exposure represents the aggregate static directional risk in the market.",
            "customer": """
**Customer Perspective:**
- **Positive DEX:** Customers are Net Long Delta (Bullish).
- **Negative DEX:** Customers are Net Short Delta (Bearish).
            """,
            "dealer": """
**Dealer Perspective:**
- **Negative DEX:** Dealers are Net Short Delta (must carry Long Stock hedge).
- **Positive DEX:** Dealers are Net Long Delta (must carry Short Stock hedge).
- **Implication:** Large Dealer positions can act as supply/demand overhangs.
            """
        },
        "Charm": {
            "title": "Charm Exposure (Delta Decay)",
            "description": "Charm measures the change in Delta due to the passage of time (dDelta/dt).",
            "customer": """
**Customer Perspective (Long Options):**
- **Positive Charm (ITM Calls / OTM Puts):** Delta increases (becomes more positive) over time.
- **Negative Charm (OTM Calls / ITM Puts):** Delta decreases (becomes more negative) over time.
            """,
            "dealer": """
**Dealer Perspective (Short Options):**
- **Negative Charm (Short ITM Calls / OTM Puts):** Dealer Short Delta becomes more negative (decreases) over time. Dealer must **BUY** flows into close.
- **Positive Charm (Short OTM Calls / ITM Puts):** Dealer Short Delta becomes less negative (increases) over time. Dealer must **SELL** flows into close.
            """
        },
        "Speed": {
            "title": "Speed Exposure (Gamma of Gamma)",
            "description": "Speed measures how quickly Gamma changes as spot price moves.",
            "customer": """
**Customer Perspective:**
- **High Speed:** Gamma changes rapidly with price.
- **Risk:** Indicates zones where instability (Gamma) can appear suddenly.
            """,
            "dealer": """
**Dealer Perspective:**
- **Hedging Risk:** High Speed means the "Gamma Trap" tightens quickly.
- **Impact:** Hedging requirements accelerate non-linearly.
            """
        },
        "Vomma": {
            "title": "Vomma Exposure (Vol of Vol)",
            "description": "Vomma measures the convexity of Vega—how sensitive Vega is to changes in IV.",
            "customer": """
**Customer Perspective:**
- **Long Vomma:** Position gains value at an accelerating rate if IV spikes.
- **Wing Risk:** Long OTM options act as "Convexity Enablers".
            """,
            "dealer": """
**Dealer Perspective:**
- **Short Vomma:** Dealer P&L suffers accelerating losses if IV spikes.
- **Blowout Risk:** Short Vomma is the primary driver of market maker stress during crashes.
            """
        },
        "Color": {
            "title": "Color Exposure (Gamma Decay)",
            "description": "Color measures the rate of change of Gamma over time.",
            "customer": """
**Customer Perspective:**
- **Near Expiry:** Gamma changes rapidly (High Color).
- **Pin Risk:** High Color indicates potential for price pinning.
            """,
            "dealer": """
**Dealer Perspective:**
- **Gamma Risk:** Measures how quickly Gamma risk escalates as expiration approaches.
- **Hedging:** High Color implies the need for rapid, high-frequency hedging adjustments.
            """
        }
    }
    
    exp = explanations.get(exposure_type, {
        "title": f"{exposure_type} Exposure",
        "description": "Exposure measurement for options positioning.",
        "customer": "Customer perspective shows long options positioning.",
        "dealer": "Dealer perspective shows short options positioning (opposite of customer)."
    })
    
    perspective_text = exp["dealer"] if is_dealer else exp["customer"]
    
    return exp["title"], exp["description"], perspective_text, perspective

def create_exposure_bar_chart(calls, puts, exposure_type, title, S):
    # Apply perspective (Dealer = Short)
    perspective = st.session_state.get('exposure_perspective', 'Customer')
    if perspective == 'Dealer':
        calls = calls.copy()
        puts = puts.copy()
        calls[exposure_type] = calls[exposure_type] * -1
        puts[exposure_type] = puts[exposure_type] * -1

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Filter out zero values
    calls_df = calls[['strike', exposure_type]].copy().fillna(0)
    calls_df = calls_df[calls_df[exposure_type] != 0]
    # Aggregate by strike to handle multiple expirations
    calls_df = calls_df.groupby('strike', as_index=False)[exposure_type].sum()
    calls_df['OptionType'] = 'Call'

    puts_df = puts[['strike', exposure_type]].copy().fillna(0)
    puts_df = puts_df[puts_df[exposure_type] != 0]
    # Aggregate by strike to handle multiple expirations
    puts_df = puts_df.groupby('strike', as_index=False)[exposure_type].sum()
    puts_df['OptionType'] = 'Put'

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Apply strike range filter
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Filter the original dataframes for net exposure calculation
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Calculate Net Exposure based on type using filtered data
    if exposure_type == 'GEX' or exposure_type == 'GEX_notional':
        if st.session_state.gex_type == 'Net':
            net_exposure = calls_filtered.groupby('strike')[exposure_type].sum().sub(puts_filtered.groupby('strike')[exposure_type].sum(), fill_value=0)
        else:  # Absolute
            calls_gex = calls_filtered.groupby('strike')[exposure_type].sum()
            puts_gex = puts_filtered.groupby('strike')[exposure_type].sum()
            # Calculate total absolute gamma exposure (Call + Put magnitudes)
            net_exposure = calls_gex.abs().add(puts_gex.abs(), fill_value=0)
    elif exposure_type == 'DEX':
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum().add(puts_filtered.groupby('strike')[exposure_type].sum(), fill_value=0)
    else:  # VEX, Charm, Speed, Vomma
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum().add(puts_filtered.groupby('strike')[exposure_type].sum(), fill_value=0)

    # Calculate total Greek values using the entire chain
    total_call_value = calls[exposure_type].fillna(0).sum()
    total_put_value = puts[exposure_type].fillna(0).sum()
    
    if exposure_type in ['GEX', 'GEX_notional']:
         total_net_value = total_call_value - total_put_value
    else:
         total_net_value = total_call_value + total_put_value

    # Get the metric being used and add it to the title
    metric_name = st.session_state.get('exposure_metric', 'Open Interest')
    delta_adjusted_label = " (Δ-Adjusted)" if st.session_state.get('delta_adjusted_exposures', False) and exposure_type != 'DEX' else ""
    notional_label = " ($)" if st.session_state.get('calculate_in_notional', True) else ""
    
    # Determine color for net value
    net_color = st.session_state.call_color if total_net_value >= 0 else st.session_state.put_color

    # Update title to include total Greek values with colored values using HTML and metric info
    title_with_totals = (
        f"{title}{delta_adjusted_label}{notional_label} ({metric_name})     "
        f"<span style='color: {st.session_state.call_color}'>{format_large_number(total_call_value)}</span> | "
        f"<span style='color: {net_color}'>Net: {format_large_number(total_net_value)}</span> | "
        f"<span style='color: {st.session_state.put_color}'>{format_large_number(total_put_value)}</span>"
    )

    fig = go.Figure()

    # Add Absolute Gamma background if enabled (GEX only)
    if exposure_type in ['GEX', 'GEX_notional'] and st.session_state.get('abs_gex_opacity', 0) > 0:
        calls_abs = calls_filtered.groupby('strike')[exposure_type].sum().abs()
        puts_abs = puts_filtered.groupby('strike')[exposure_type].sum().abs()
        total_abs_gex = calls_abs.add(puts_abs, fill_value=0)
        
        opacity = st.session_state.abs_gex_opacity
        fill_color = f'rgba(128, 128, 128, {opacity})'
        
        if st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Scatter(
                y=total_abs_gex.index,
                x=total_abs_gex.values,
                mode='lines',
                fill='tozerox',
                name='Abs Gamma',
                line=dict(color='rgba(255, 255, 255, 0)', width=0),
                fillcolor=fill_color,
                hoverinfo='skip',
                orientation='h',
                showlegend=False
            ))
        else:
             fig.add_trace(go.Scatter(
                x=total_abs_gex.index,
                y=total_abs_gex.values,
                mode='lines',
                fill='tozeroy',
                name='Abs Gamma',
                line=dict(color='rgba(255, 255, 255, 0)', width=0),
                fillcolor=fill_color,
                hoverinfo='skip',
                showlegend=False
            ))

    # Add calls if enabled
    if (st.session_state.show_calls):
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=calls_df['strike'],
                x=calls_df[exposure_type],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=puts_df['strike'],
                x=puts_df[exposure_type],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_exposure.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_exposure.index,
                y=net_exposure.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_exposure.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=net_exposure.index,
                x=net_exposure.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_exposure.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding
    y_values = []
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_range = y_max - y_min
        
        # Ensure minimum range and add padding
        if abs(y_range) < 1:
            y_range = 1
        
        # Add 15% padding on top and bottom
        padding = y_range * 0.15
        y_min = y_min - padding
        y_max = y_max + padding
    else:
        # Default values if no valid y values
        y_min = -1
        y_max = 1

    # Update layout with calculated y-range
    padding = strike_range * 0.1
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                xref="paper",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)  # Title slightly larger
            ),
            xaxis_title=dict(
                text=f"{title} ({metric_name})",
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increase chart height for better visibility
        )
    else:
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                xref="paper",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)  # Title slightly larger
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text=f"{title} ({metric_name})",
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increase chart height for better visibility
        )

    # Logic to fix visual gaps by using categorical axis while maintaining current price line
    # Calculate all strikes present in the chart data (filtered by strike range)
    present_strikes = sorted(list(set(calls_df['strike']) | set(puts_df['strike']) | set(net_exposure.index if not net_exposure.empty else [])))
    
    if present_strikes:
        # Calculate nice ticks to emulate linear axis labeling
        tick_vals = present_strikes
        if len(present_strikes) > 15:
            try:
                min_s, max_s = min(present_strikes), max(present_strikes)
                # Target around 10 ticks (adjusted for chart width/height)
                target_ticks = 10
                raw_step = (max_s - min_s) / target_ticks
                # Find nearest nice step (1, 2, 5, 10, etc)
                mag = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
                residual = raw_step / mag
                if residual >= 5: nice_step = 5 * mag
                elif residual >= 2: nice_step = 2 * mag
                else: nice_step = mag
                
                nice_ticks = []
                # Start from a clean multiple
                curr = math.ceil(min_s / nice_step) * nice_step
                while curr <= max_s + (nice_step * 0.1): # Add small buffer for float comparison
                    # Find closest strike in data to this nice number
                    # We only show a tick if there is a strike reasonably close to the nice number
                    closest = min(present_strikes, key=lambda x: abs(x - curr))
                    # Only add if it's unique and reasonably close (within 50% of step)
                    if closest not in nice_ticks and abs(closest - curr) < nice_step * 0.5:
                        nice_ticks.append(closest)
                    curr += nice_step
                tick_vals = sorted(nice_ticks)
            except Exception:
                tick_vals = present_strikes # Fallback

        # Interpolate S position among present strikes
        spot_pos = 0
        if S <= present_strikes[0]:
            spot_pos = -0.5
        elif S >= present_strikes[-1]:
            spot_pos = len(present_strikes) - 0.5
        else:
            for i in range(len(present_strikes)-1):
                if present_strikes[i] <= S <= present_strikes[i+1]:
                    ratio = (S - present_strikes[i]) / (present_strikes[i+1] - present_strikes[i])
                    spot_pos = i + ratio
                    break
        
        # Apply categorical axis and add price line based on calculated position
        if st.session_state.chart_type == 'Horizontal Bar':
            fig.update_layout(yaxis=dict(
                type='category', 
                categoryorder='array', 
                categoryarray=present_strikes,
                tickmode='array',
                tickvals=tick_vals
            ))
            fig.add_hline(
                y=spot_pos,
                line_dash="dash",
                line_color="white",
                opacity=0.7
            )
        else:
            fig.update_layout(xaxis=dict(
                type='category', 
                categoryorder='array', 
                categoryarray=present_strikes,
                tickmode='array',
                tickvals=tick_vals
            ))
            fig.add_vline(
                x=spot_pos,
                line_dash="dash",
                line_color="white",
                opacity=0.7,
                annotation_text=f"{S}",
                annotation_position="top",
                annotation=dict(
                    font=dict(size=st.session_state.chart_text_size)
                )
            )
    else:
        # Fallback for empty data
        fig = add_current_price_line(fig, S)

    return fig

def calculate_max_pain(calls, puts):
    """Calculate max pain points based on call and put options."""
    if calls.empty or puts.empty:
        return None, None, None, None, None

    unique_strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))
    total_pain_by_strike = {}
    call_pain_by_strike = {}
    put_pain_by_strike = {}

    for strike in unique_strikes:
        # Calculate call pain (loss to option writers)
        call_subset = calls[calls['strike'] <= strike]
        call_pain = call_subset['openInterest'].fillna(0) * (strike - call_subset['strike'])
        call_pain_sum = call_pain.sum()
        call_pain_by_strike[strike] = call_pain_sum
        
        # Calculate put pain (loss to option writers)
        put_subset = puts[puts['strike'] >= strike]
        put_pain = put_subset['openInterest'].fillna(0) * (put_subset['strike'] - strike)
        put_pain_sum = put_pain.sum()
        put_pain_by_strike[strike] = put_pain_sum
        
        total_pain_by_strike[strike] = call_pain_sum + put_pain_sum

    if not total_pain_by_strike:
        return None, None, None, None, None

    max_pain_strike = min(total_pain_by_strike.items(), key=lambda x: x[1])[0]
    call_max_pain_strike = min(call_pain_by_strike.items(), key=lambda x: x[1])[0]
    put_max_pain_strike = min(put_pain_by_strike.items(), key=lambda x: x[1])[0]
    
    return (max_pain_strike, call_max_pain_strike, put_max_pain_strike, 
            total_pain_by_strike, call_pain_by_strike, put_pain_by_strike)

def create_max_pain_chart(calls, puts, S, date_count=1):
    """Create a chart showing max pain analysis with separate call and put pain."""
    result = calculate_max_pain(calls, puts)
    if result is None:
        return None
    
    (max_pain_strike, call_max_pain_strike, put_max_pain_strike,
     total_pain_by_strike, call_pain_by_strike, put_pain_by_strike) = result

    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    padding = strike_range * 0.1
    
    fig = go.Figure()

    # Add total pain line (always vertical for max pain chart)
    # Use a neutral color that works well with both call and put colors
    total_pain_color = '#FFD700'  # Gold color for total pain
    
    if st.session_state.chart_type in ['Bar', 'Horizontal Bar']:
        fig.add_trace(go.Bar(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            name='Total Pain',
            marker_color=total_pain_color
        ))
    elif st.session_state.chart_type == 'Line':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='lines',
            name='Total Pain',
            line=dict(color=total_pain_color, width=2)
        ))
    elif st.session_state.chart_type == 'Scatter':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='markers',
            name='Total Pain',
            marker=dict(color=total_pain_color)
        ))
    else:  # Area
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            fill='tozeroy',
            name='Total Pain',
            line=dict(color=total_pain_color, width=0.5),
            fillcolor='rgba(255, 215, 0, 0.3)'  # Semi-transparent gold
        ))

    # Add call pain line
    if st.session_state.show_calls:
        fig.add_trace(go.Scatter(
            x=list(call_pain_by_strike.keys()),
            y=list(call_pain_by_strike.values()),
            name='Call Pain',
            line=dict(color=call_color, width=1, dash='dot')
        ))

    # Add put pain line
    if st.session_state.show_puts:
        fig.add_trace(go.Scatter(
            x=list(put_pain_by_strike.keys()),
            y=list(put_pain_by_strike.values()),
            name='Put Pain',
            line=dict(color=put_color, width=1, dash='dot')
        ))

    # Calculate y-axis range with improved padding
    y_values = []
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_range = y_max - y_min
        
        # Add 15% padding on top and 5% on bottom (unless y_min is 0)
        padding_top = y_range * 0.15
        padding_bottom = y_range * 0.05 if y_min > 0 else 0
        y_min = y_min - padding_bottom
        y_max = y_max + padding_top
    else:
        # Default values if no valid y values
        y_min = 0
        y_max = 100

    # Add vertical lines for different max pain points
    fig.add_vline(
        x=max_pain_strike,
        line_dash="dash",
        line_color=total_pain_color,
        opacity=0.7,
        annotation_text=f"Total Max Pain: {max_pain_strike}",
        annotation_position="top"
    )

    if st.session_state.show_calls:
        fig.add_vline(
            x=call_max_pain_strike,
            line_dash="dash",
            line_color=call_color,
            opacity=0.7,
            annotation_text=f"Call Max Pain: {call_max_pain_strike}",
            annotation_position="top"
        )

    if st.session_state.show_puts:
        fig.add_vline(
            x=put_max_pain_strike,
            line_dash="dash",
            line_color=put_color,
            opacity=0.7,
            annotation_text=f"Put Max Pain: {put_max_pain_strike}",
            annotation_position="top"
        )

    # Add current price line
    fig.add_vline(
        x=S,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"{S}",
        annotation_position="bottom"
    )

    date_suffix = f" ({date_count} dates)" if date_count > 1 else ""

    fig.update_layout(
        title=dict(
            text=f'Max Pain{date_suffix}',
            font=dict(size=st.session_state.chart_text_size + 3)  # Title slightly larger
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Total Pain',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        hovermode='x unified',
        xaxis=dict(
            autorange=True,
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            autorange=True,
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        height=600  # Increased height for better visibility
    )
    
    # Remove range slider for max pain chart as requested
    fig.update_xaxes(rangeslider=dict(visible=False))
    
    return fig

@st.cache_data(ttl=get_cache_ttl(), show_spinner=False)  # Cache TTL matches refresh rate
def get_nearest_expiry(available_dates):
    """Get the nearest expiry date from a list of available dates"""
    if not available_dates:
        return None
    
    today = get_now_et().date()
    future_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in available_dates if datetime.strptime(date, '%Y-%m-%d').date() >= today]
    
    if not future_dates:
        return None
    
    return min(future_dates).strftime('%Y-%m-%d')

def create_davi_chart(calls, puts, S, date_count=1):
    """Create Delta-Adjusted Value Index chart that matches other exposure charts style"""
    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Create deep copies to avoid modifying original dataframes
    calls_df = calls.copy()
    puts_df = puts.copy()
    
    # Check if calc_delta column exists, if not, calculate delta
    if 'calc_delta' not in calls_df.columns:
        # Get current date and calculate time to expiration
        today = get_now_et().date()
        
        # Extract expiry date - use the first one if multiple
        if 'extracted_expiry' in calls_df.columns and not calls_df['extracted_expiry'].empty:
            selected_expiry = calls_df['extracted_expiry'].iloc[0]
            
            # Calculate time to expiration more precisely
            t = calculate_time_to_expiration(selected_expiry)
            t = max(t, 1e-5)
            
            # Define function to compute delta
            def compute_delta(row, flag):
                sigma = row.get("impliedVolatility", None)
                if sigma is None or sigma <= 0:
                    return 0.5  # Default delta if IV is missing or invalid
                try:
                    delta_val, _, _ = calculate_greeks(flag, S, row["strike"], t, sigma)
                    return delta_val
                except Exception:
                    return 0.5  # Default delta if calculation fails
            
            # Calculate delta for calls and puts
            calls_df["calc_delta"] = calls_df.apply(lambda row: compute_delta(row, "c"), axis=1)
            puts_df["calc_delta"] = puts_df.apply(lambda row: compute_delta(row, "p"), axis=1)
        else:
            # If no expiry information, use approximate delta based on strike
            calls_df["calc_delta"] = calls_df.apply(lambda row: max(0, min(1, 1 - (row["strike"] - S) / (S * 0.1))), axis=1)
            puts_df["calc_delta"] = puts_df.apply(lambda row: max(0, min(1, (row["strike"] - S) / (S * 0.1))), axis=1)
    
    # Determine which metric to use
    metric_type = st.session_state.get('exposure_metric', 'Open Interest')
    
    if metric_type == 'Volume':
        calls_metric = calls_df['volume'].fillna(0)
        puts_metric = puts_df['volume'].fillna(0)
    elif metric_type == 'Volume Weighted by OI':
        calls_metric = (calls_df['volume'].fillna(0) + calls_df['openInterest'].fillna(0)) / 2
        puts_metric = (puts_df['volume'].fillna(0) + puts_df['openInterest'].fillna(0)) / 2
    else: # Open Interest
        calls_metric = calls_df['openInterest'].fillna(0)
        puts_metric = puts_df['openInterest'].fillna(0)

    # Calculate DAVI for calls and puts with filtering
    # Only keep non-zero values
    calls_mid_price = (calls_df['bid'].fillna(0) + calls_df['ask'].fillna(0)) / 2
    calls_df['DAVI'] = calls_metric * 100 * calls_mid_price * calls_df['calc_delta'].fillna(0)
    calls_df = calls_df[calls_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    # Aggregate by strike to handle multiple expirations
    calls_df = calls_df.groupby('strike', as_index=False)['DAVI'].sum()
    calls_df['OptionType'] = 'Call'

    puts_mid_price = (puts_df['bid'].fillna(0) + puts_df['ask'].fillna(0)) / 2
    puts_df['DAVI'] = puts_metric * 100 * puts_mid_price * puts_df['calc_delta'].fillna(0)
    puts_df = puts_df[puts_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    # Aggregate by strike to handle multiple expirations
    puts_df = puts_df.groupby('strike', as_index=False)['DAVI'].sum()
    puts_df['OptionType'] = 'Put'

    # Apply perspective (Dealer = Short, flip the sign)
    perspective = st.session_state.get('exposure_perspective', 'Customer')
    if perspective == 'Dealer':
        calls_df['DAVI'] = calls_df['DAVI'] * -1
        puts_df['DAVI'] = puts_df['DAVI'] * -1

    # Calculate totals for title using the entire chain (before filtering by strike range)
    total_call_davi = calls_df['DAVI'].sum()
    total_put_davi = puts_df['DAVI'].sum()
    total_net_davi = total_call_davi + total_put_davi

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Calculate Net DAVI
    net_davi = pd.Series(0, index=sorted(set(calls_df['strike']) | set(puts_df['strike'])))
    if not calls_df.empty:
        net_davi = net_davi.add(calls_df.groupby('strike')['DAVI'].sum(), fill_value=0)
    if not puts_df.empty:
        net_davi = net_davi.add(puts_df.groupby('strike')['DAVI'].sum(), fill_value=0)

    date_suffix = f" ({date_count} dates)" if date_count > 1 else ""

    # Determine color for net value
    net_davi_color = call_color if total_net_davi >= 0 else put_color

    # Create title with totals
    metric_name = metric_type
    title_with_totals = (
        f"Delta-Adjusted Value Index ({metric_name}) by Strike{date_suffix}     "
        f"<span style='color: {call_color}'>{format_large_number(total_call_davi)}</span> | "
        f"<span style='color: {net_davi_color}'>Net: {format_large_number(total_net_davi)}</span> | "
        f"<span style='color: {put_color}'>{format_large_number(total_put_davi)}</span>"
    )

    fig = go.Figure()

    # Add calls if enabled
    if st.session_state.show_calls:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=calls_df['strike'],
                x=calls_df['DAVI'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=puts_df['strike'],
                x=puts_df['DAVI'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_davi.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_davi.index,
                y=net_davi.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_davi.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=net_davi.index,
                x=net_davi.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_davi.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Add current price line
    fig = add_current_price_line(fig, S)

    # Update layout
    padding = strike_range * 0.1
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='DAVI',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='y unified',
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increased height for better visibility
        )
    else:
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 3)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='DAVI',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increased height for better visibility
        )

    return fig

# Add at the start of each page's container
if st.session_state.current_page:
    market_status = check_market_status()
    if market_status:
        st.warning(market_status)

# Create main placeholder for page content
main_placeholder = st.empty()

if st.session_state.current_page == "OI & Volume":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="options_data_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_oi"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache and expiry selections if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
            
            # Clear expiry selection state for current page when ticker changes
            page_expiry_key = f"{st.session_state.current_page}_selected_dates"
            if page_expiry_key in st.session_state:
                st.session_state[page_expiry_key] = []
            
            # Also clear any expiry selector widgets
            selector_key = f"{st.session_state.current_page}_expiry_selector"
            if selector_key in st.session_state:
                st.session_state[selector_key] = []
            
            # Force rerun to refresh available expiry dates
            st.rerun()
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.info("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                # New: Add tabs to organize content
                tab1, tab2, tab3, tab4 = st.tabs(["OI & Volume Charts", "Options Flow Analysis", "Premium Analysis", "Market Maker"])
                
                with tab1:
                    # Calculate totals for header display
                    total_call_oi = all_calls['openInterest'].sum()
                    total_put_oi = all_puts['openInterest'].sum()
                    total_call_vol = all_calls['volume'].sum()
                    total_put_vol = all_puts['volume'].sum()
                    
                    # Display colorful metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"**Total Call OI**")
                        st.markdown(f"<span style='color:{st.session_state.call_color}; font-size: 20px'>{format_large_number(total_call_oi)}</span>", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"**Total Put OI**")
                        st.markdown(f"<span style='color:{st.session_state.put_color}; font-size: 20px'>{format_large_number(total_put_oi)}</span>", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"**Total Call Vol**")
                        st.markdown(f"<span style='color:{st.session_state.call_color}; font-size: 20px'>{format_large_number(total_call_vol)}</span>", unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"**Total Put Vol**")
                        st.markdown(f"<span style='color:{st.session_state.put_color}; font-size: 20px'>{format_large_number(total_put_vol)}</span>", unsafe_allow_html=True)
                        
                    st.markdown("---")

                    # Original OI and Volume charts
                    oi_fig, volume_fig = create_oi_volume_charts(all_calls, all_puts, S, len(selected_expiry_dates))
                    st.plotly_chart(oi_fig, width='stretch')
                    st.plotly_chart(volume_fig, width='stretch')
                
                with tab2:
                    # New: Options flow analysis and visualizations
                    flow_data = analyze_options_flow(all_calls, all_puts, S)
                    # Note: create_option_flow_charts returns: fig_volume, fig_premium, fig_itm_otm_vol, fig_itm_otm_prem
                    fig_volume_chart, fig_premium_chart, fig_itm_otm_vol, fig_itm_otm_prem = create_option_flow_charts(flow_data)
                    
                    st.subheader("Options Flow Overview")
                    
                    col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
                    with col_metrics_1:
                         st.markdown(f"**Total Call Volume**")
                         st.markdown(f"<span style='color:{st.session_state.call_color}; font-size: 20px'>{flow_data['calls']['volume']:,.0f}</span>", unsafe_allow_html=True)
                         st.markdown(f"**Total Put Volume**")
                         st.markdown(f"<span style='color:{st.session_state.put_color}; font-size: 20px'>{flow_data['puts']['volume']:,.0f}</span>", unsafe_allow_html=True)
                    with col_metrics_2:
                         st.markdown(f"**Total Call Premium**")
                         st.markdown(f"<span style='color:{st.session_state.call_color}; font-size: 20px'>${flow_data['total_premium']['calls']:,.0f}</span>", unsafe_allow_html=True)
                         st.markdown(f"**Total Put Premium**")
                         st.markdown(f"<span style='color:{st.session_state.put_color}; font-size: 20px'>${flow_data['total_premium']['puts']:,.0f}</span>", unsafe_allow_html=True)
                    with col_metrics_3:
                        pcr_vol = flow_data['puts']['volume'] / max(flow_data['calls']['volume'], 1)
                        pcr_prem = flow_data['total_premium']['puts'] / max(flow_data['total_premium']['calls'], 1)
                        
                        pcr_vol_color = st.session_state.call_color if pcr_vol < 1 else st.session_state.put_color if pcr_vol > 1 else 'white'
                        pcr_prem_color = st.session_state.call_color if pcr_prem < 1 else st.session_state.put_color if pcr_prem > 1 else 'white'

                        st.markdown(f"**Volume PCR**")
                        st.markdown(f"<span style='color:{pcr_vol_color}; font-size: 20px'>{pcr_vol:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Premium PCR**")
                        st.markdown(f"<span style='color:{pcr_prem_color}; font-size: 20px'>{pcr_prem:.2f}</span>", unsafe_allow_html=True)

                    # Create two columns for charts
                    flow_col1, flow_col2 = st.columns(2)
                    
                    with flow_col1:
                        st.plotly_chart(fig_volume_chart, width='stretch')
                        st.plotly_chart(fig_itm_otm_vol, width='stretch')
                    
                    with flow_col2:
                        st.plotly_chart(fig_premium_chart, width='stretch')
                        st.plotly_chart(fig_itm_otm_prem, width='stretch')
                    
                    # Summary metrics display
                    st.subheader("Flow Detail Summary")
                    
                    summary_data = {
                        'Metric': [
                            'Total Premium', 'Volume', 'ITM Volume', 'OTM Volume', 'ITM Premium', 'OTM Premium'
                        ],
                        'Calls': [
                            f"${flow_data['total_premium']['calls']:,.0f}",
                            f"{flow_data['calls']['volume']:,.0f}",
                            f"{flow_data['calls']['ITM']['volume']:,.0f}",
                            f"{flow_data['calls']['OTM']['volume']:,.0f}",
                            f"${flow_data['calls']['ITM']['premium']:,.0f}",
                            f"${flow_data['calls']['OTM']['premium']:,.0f}"
                        ],
                        'Puts': [
                            f"${flow_data['total_premium']['puts']:,.0f}",
                            f"{flow_data['puts']['volume']:,.0f}",
                            f"{flow_data['puts']['ITM']['volume']:,.0f}",
                            f"{flow_data['puts']['OTM']['volume']:,.0f}",
                            f"${flow_data['puts']['ITM']['premium']:,.0f}",
                            f"${flow_data['puts']['OTM']['premium']:,.0f}"
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Apply coloring to the dataframe
                    st.dataframe(
                        summary_df.style.map(
                            lambda x: f'color: {st.session_state.call_color}; font-weight: bold', subset=['Calls']
                        ).map(
                            lambda x: f'color: {st.session_state.put_color}; font-weight: bold', subset=['Puts']
                        ), 
                        hide_index=True, 
                        width='stretch'
                    )
                
                with tab3:
                    # New: Advanced premium analysis
                    st.subheader("Premium Distribution Analysis")
                    
                    
                    # Premium summary statistics
                    # Calculate Mid Price for better accuracy
                    all_calls['midPrice'] = (all_calls['bid'].fillna(0) + all_calls['ask'].fillna(0)) / 2
                    all_puts['midPrice'] = (all_puts['bid'].fillna(0) + all_puts['ask'].fillna(0)) / 2

                    total_call_premium = (all_calls['volume'] * all_calls['midPrice'] * 100).sum()
                    total_put_premium = (all_puts['volume'] * all_puts['midPrice'] * 100).sum()
                    premium_ratio = total_call_premium / max(total_put_premium, 1)  # Avoid division by zero
                    
                    # Premium by moneyness
                    all_calls['moneyness'] = all_calls.apply(lambda x: 'ITM' if x['strike'] <= S else 'OTM', axis=1)
                    all_puts['moneyness'] = all_puts.apply(lambda x: 'ITM' if x['strike'] >= S else 'OTM', axis=1)
                    
                    otm_call_premium = (all_calls[all_calls['moneyness'] == 'OTM']['volume'] * 
                                    all_calls[all_calls['moneyness'] == 'OTM']['midPrice'] * 100).sum()
                    itm_call_premium = (all_calls[all_calls['moneyness'] == 'ITM']['volume'] * 
                                    all_calls[all_calls['moneyness'] == 'ITM']['midPrice'] * 100).sum()
                    otm_put_premium = (all_puts[all_puts['moneyness'] == 'OTM']['volume'] * 
                                    all_puts[all_puts['moneyness'] == 'OTM']['midPrice'] * 100).sum()
                    itm_put_premium = (all_puts[all_puts['moneyness'] == 'ITM']['volume'] * 
                                    all_puts[all_puts['moneyness'] == 'ITM']['midPrice'] * 100).sum()
                    
                    # Calculate ITM premium flow
                    itm_net_premium = itm_call_premium - itm_put_premium
                    itm_premium_ratio = itm_call_premium / max(itm_put_premium, 1)
                    
                    # Determine ITM premium flow sentiment
                    if itm_premium_ratio > 1.5:
                        itm_sentiment = "Bullish"
                        itm_color = st.session_state.call_color
                    elif itm_premium_ratio < 0.7:
                        itm_sentiment = "Bearish"
                        itm_color = st.session_state.put_color
                    else:
                        itm_sentiment = "Neutral"
                        itm_color = "white"
                        
                    # Display premium metrics in a cleaner format with call/put ratio indicator
                    st.markdown("### Premium Summary")
                    
                    # Call/put premium ratio status indicator
                    ratio_status = ""
                    if premium_ratio > 1.5:
                        ratio_status = "Bullish (high call premium)"
                        ratio_color = st.session_state.call_color
                    elif premium_ratio < 0.7:
                        ratio_status = "Bearish (high put premium)"
                        ratio_color = st.session_state.put_color
                    else:
                        ratio_status = "Neutral"
                        ratio_color = "white"
                    
                    # Create metrics with custom styling
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(50,50,50,0.3); margin-bottom: 15px;">
                            <h4>Call/Put Premium Ratio: <span style="color: {ratio_color}">{premium_ratio:.2f}</span> {ratio_status}</h4>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Add ITM premium flow indicator
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(50,50,50,0.3); margin-bottom: 15px;">
                            <h4>ITM Premium Flow: <span style="color: {itm_color}">{itm_sentiment}</span> (Call/Put Ratio: {itm_premium_ratio:.2f})</h4>
                            <p>Net ITM Premium: <span style="color: {st.session_state.call_color if itm_net_premium > 0 else st.session_state.put_color}">${abs(itm_net_premium):,.0f}</span> 
                            {" toward calls" if itm_net_premium > 0 else " toward puts"}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Create three columns for better organization
                    premium_col1, premium_col2, premium_col3 = st.columns(3)
                    
                    with premium_col1:
                        st.markdown(f"<h4 style='color: {st.session_state.call_color}'>Call Premium</h4>", unsafe_allow_html=True)
                        st.metric("Total", f"${total_call_premium:,.0f}")
                        st.metric("OTM", f"${otm_call_premium:,.0f}", f"{otm_call_premium/total_call_premium*100:.1f}%" if total_call_premium > 0 else "0%")
                        st.metric("ITM", f"${itm_call_premium:,.0f}", f"{itm_call_premium/total_call_premium*100:.1f}%" if total_call_premium > 0 else "0%")
                    
                    with premium_col2:
                        st.markdown(f"<h4 style='color: {st.session_state.put_color}'>Put Premium</h4>", unsafe_allow_html=True)
                        st.metric("Total", f"${total_put_premium:,.0f}")
                        st.metric("OTM", f"${otm_put_premium:,.0f}", f"{otm_put_premium/total_put_premium*100:.1f}%" if total_put_premium > 0 else "0%")
                        st.metric("ITM", f"${itm_put_premium:,.0f}", f"{itm_put_premium/total_put_premium*100:.1f}%" if total_put_premium > 0 else "0%")
                    
                    with premium_col3:
                        st.markdown("<h4>Premium Analysis</h4>", unsafe_allow_html=True)
                        
                        # Calculate OTM/ITM ratios for sentiment analysis
                        otm_itm_call_ratio = otm_call_premium / max(itm_call_premium, 1)
                        otm_itm_put_ratio = otm_put_premium / max(itm_put_premium, 1)
                        
                        # Show OTM to ITM ratios
                        st.metric("OTM/ITM Call Ratio", f"{otm_itm_call_ratio:.2f}")
                        st.metric("OTM/ITM Put Ratio", f"{otm_itm_put_ratio:.2f}")
                        
                        # Premium concentration
                        total_premium = total_call_premium + total_put_premium
                        st.metric("Call Premium %", f"{total_call_premium/total_premium*100:.1f}%" if total_premium > 0 else "0%")
                        st.metric("Put Premium %", f"{total_put_premium/total_premium*100:.1f}%" if total_premium > 0 else "0%")
                        
                    # Add new ITM premium flow analysis chart
                    st.markdown("### ITM Premium Flow Analysis")
                    
                    # Create a bar chart to visualize ITM premium distribution
                    itm_premium_fig = go.Figure()
                    
                    # Add bars for ITM Call and Put premium
                    itm_premium_fig.add_trace(go.Bar(
                        x=['ITM Calls'],
                        y=[itm_call_premium],
                        name='ITM Call Premium',
                        marker_color=st.session_state.call_color
                    ))
                    
                    itm_premium_fig.add_trace(go.Bar(
                        x=['ITM Puts'],
                        y=[itm_put_premium],
                        name='ITM Put Premium',
                        marker_color=st.session_state.put_color
                    ))
                    
                    # Add a third bar for net ITM premium flow if needed
                    itm_premium_fig.add_trace(go.Bar(
                        x=['Net ITM Flow'],
                        y=[itm_net_premium],
                        name='Net ITM Premium',
                        marker_color=st.session_state.call_color if itm_net_premium > 0 else st.session_state.put_color
                    ))
                    
                    itm_premium_fig.update_layout(
                        title=dict(
                            text=f"ITM Premium Flow - {itm_sentiment}",
                            font=dict(size=st.session_state.chart_text_size + 6)
                        ),
                        xaxis_title=dict(
                            text="Option Type",
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis_title=dict(
                            text="Premium ($)",
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        template="plotly_dark",
                        xaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size)
                        )
                    )
                    
                    st.plotly_chart(itm_premium_fig, width='stretch')
                    
                    # Add additional premium insights with ITM flow details
                    st.markdown("### Premium Insights")
                    
                    # Create strike-based premium insights - Fix for deprecation warning
                    # Instead of using groupby().apply(), calculate premium directly
                    calls_premium = all_calls.copy()
                    calls_premium['premium'] = calls_premium['volume'] * calls_premium['midPrice'] * 100
                    call_premium_by_strike = calls_premium.groupby('strike')['premium'].sum().reset_index()
                    
                    puts_premium = all_puts.copy()
                    puts_premium['premium'] = puts_premium['volume'] * puts_premium['midPrice'] * 100
                    put_premium_by_strike = puts_premium.groupby('strike')['premium'].sum().reset_index()
                    
                    # Find top premium concentrations
                    top_call_strikes = call_premium_by_strike.nlargest(5, 'premium')
                    top_put_strikes = put_premium_by_strike.nlargest(5, 'premium')
                    
                    # Calculate call vs put premium for each strike and net premium flow
                    premium_combined = pd.merge(call_premium_by_strike, put_premium_by_strike, on='strike', how='outer', suffixes=('_call', '_put')).fillna(0)
                    premium_combined['net_premium'] = premium_combined['premium_call'] - premium_combined['premium_put']
                    premium_combined['ratio'] = premium_combined['premium_call'] / premium_combined['premium_put'].replace(0, 1)
                    
                    # Find strikes with most bullish and bearish premium flow
                    bullish_strikes = premium_combined.nlargest(5, 'net_premium')
                    bearish_strikes = premium_combined.nsmallest(5, 'net_premium')
                    
                    # Calculate total premium for percentages
                    total_call_premium_sum = call_premium_by_strike['premium'].sum()
                    total_put_premium_sum = put_premium_by_strike['premium'].sum()
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    # Use a table format instead of text lines to avoid formatting issues
                    with insight_col1:
                        st.markdown(f"<h5 style='color: {st.session_state.call_color}'>Top Call Premium Strikes</h5>", unsafe_allow_html=True)
                        
                        # Create DataFrames for display
                        call_strikes_data = []
                        for _, row in top_call_strikes.iterrows():
                            pct = (row['premium'] / total_call_premium_sum * 100) if total_call_premium_sum > 0 else 0
                            call_strikes_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Premium": f"${row['premium']:,.0f}", 
                                "% of Total": f"{pct:.1f}%"
                            })
                        st.table(pd.DataFrame(call_strikes_data))
                        
                        st.markdown(f"<h5 style='color: {st.session_state.call_color}'>Most Bullish Premium Flow</h5>", unsafe_allow_html=True)
                        
                        bullish_data = []
                        for _, row in bullish_strikes.iterrows():
                            bullish_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Net Premium": f"+${row['net_premium']:,.0f}", 
                                "C/P Ratio": f"{row['ratio']:.2f}"
                            })
                        st.table(pd.DataFrame(bullish_data))
                    
                    with insight_col2:
                        st.markdown(f"<h5 style='color: {st.session_state.put_color}'>Top Put Premium Strikes</h5>", unsafe_allow_html=True)
                        
                        put_strikes_data = []
                        for _, row in top_put_strikes.iterrows():
                            pct = (row['premium'] / total_put_premium_sum * 100) if total_put_premium_sum > 0 else 0
                            put_strikes_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Premium": f"${row['premium']:,.0f}", 
                                "% of Total": f"{pct:.1f}%"
                            })
                        st.table(pd.DataFrame(put_strikes_data))
                        
                        st.markdown(f"<h5 style='color: {st.session_state.put_color}'>Most Bearish Premium Flow</h5>", unsafe_allow_html=True)
                        
                        bearish_data = []
                        for _, row in bearish_strikes.iterrows():
                            bearish_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Net Premium": f"-${abs(row['net_premium']):,.0f}", 
                                "C/P Ratio": f"{row['ratio']:.2f}"
                            })
                        st.table(pd.DataFrame(bearish_data))

                with tab4:
                    # Market Maker tab content
                    st.write("### 📊 Market Maker Positioning")
                    st.write(f"**Symbol:** {ticker.upper()} | **Expiry:** {', '.join(selected_expiry_dates)}")
                    st.info("📅 **Data Source**: OCC (Options Clearing Corporation) | **Timing**: Latest business day from past 24 hours")
                    
                    # Add informational section about OCC and data timing
                    with st.expander("ℹ️ About OCC Market Maker Data", expanded=False):
                        st.markdown("""
                        **What is the OCC (Options Clearing Corporation)?**
                        
                        The OCC is the world's largest equity derivatives clearing organization and acts as the central counterparty for all options trades in the U.S. They guarantee the performance of all options contracts and provide transparency into market activity.
                        
                        **Market Maker Role:**
                        - Market makers provide liquidity by continuously quoting bid and ask prices
                        - They facilitate trading by being ready to buy or sell options contracts
                        - Their positioning data reveals institutional sentiment and flow direction
                        
                        **Data Timing & Availability:**
                        - 📅 **Report Date**: Data is from the latest business day within the past 24 hours
                        - ⏰ **Update Schedule**: OCC updates this data daily after market close
                        - 🕐 **Lag Time**: Data typically reflects previous trading day activity
                        - 📊 **Coverage**: All option types (equity, index, ETF options)
                        
                        **Why This Matters:**
                        - Market maker positioning can indicate institutional sentiment
                        - Large call positions may suggest bullish positioning
                        - Large put positions may suggest bearish positioning or hedging activity
                        - Combined with other analysis, helps understand market dynamics
                        
                        **Important Notes:**
                        - This is historical data (not real-time)
                        - Market maker positions can change rapidly during trading hours
                        - Data should be used in conjunction with other analysis tools
                        """)
                    
                    st.write("")  # Add spacing
                    
                    # Initialize session state for market maker data
                    if 'mm_data' not in st.session_state:
                        st.session_state.mm_data = None
                        st.session_state.mm_message = None
                        st.session_state.mm_last_ticker = None
                        st.session_state.mm_last_expiry = None
                    
                    # Check if we need to fetch new data (ticker or expiry changed)
                    current_expiries = tuple(selected_expiry_dates) if selected_expiry_dates else None
                    need_refresh = (
                        st.session_state.mm_last_ticker != ticker or 
                        st.session_state.mm_last_expiry != current_expiries or
                        st.session_state.mm_data is None
                    )
                    
                    # Auto-fetch data when symbol or expiry changes
                    if need_refresh and ticker and current_expiries:
                        with st.spinner("Loading market maker data for multiple expiration dates..."):
                            combined_data = []
                            success_messages = []
                            
                            # Capture context for threads
                            ctx = get_script_run_ctx() if get_script_run_ctx else None

                            def fetch_mm_data(expiry_date):
                                if add_script_run_ctx and ctx:
                                    add_script_run_ctx(threading.current_thread(), ctx)
                                return download_volume_csv(ticker, "U", expiry_date), expiry_date

                            # Fetch data for each selected expiration date in parallel
                            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                future_to_date = {}
                                for date in selected_expiry_dates:
                                    future = executor.submit(fetch_mm_data, date)
                                    future_to_date[future] = date
                                
                                # Collect results as they complete, but we want to maintain order or just collect all
                                # Since order doesn't strictly matter for combination, we can just collect
                                results = []
                                for future in concurrent.futures.as_completed(future_to_date):
                                    try:
                                        (data, message), date = future.result()
                                        results.append((date, data, message))
                                    except Exception as e:
                                        results.append((future_to_date[future], None, f"Error: {e}"))
                            
                            # Sort results by date to keep messages consistent
                            results.sort(key=lambda x: x[0])
                            
                            for date, data, message in results:
                                if data:
                                    combined_data.append(data)
                                    success_messages.append(f"✓ {date}")
                                else:
                                    success_messages.append(f"✗ {date}: {message}")
                            
                            # Combine all CSV data
                            if combined_data:
                                # Join all CSV data with headers
                                all_csv_data = []
                                for i, csv_data in enumerate(combined_data):
                                    lines = csv_data.strip().split('\n')
                                    # Skip header lines and add only data lines
                                    data_lines = []
                                    for line in lines:
                                        if ',' in line and line.strip() and line.strip()[0].isdigit():
                                            data_lines.extend(lines[lines.index(line):])
                                            break
                                    all_csv_data.extend(data_lines)
                                
                                combined_csv = '\n'.join(all_csv_data)
                                combined_message = f"Market Maker Data Retrieved for {len(selected_expiry_dates)} expiration dates:\n" + '\n'.join(success_messages)
                                
                                st.session_state.mm_data = combined_csv
                                st.session_state.mm_message = combined_message
                            else:
                                st.session_state.mm_data = None
                                st.session_state.mm_message = "Failed to retrieve data for any selected expiration dates."
                            
                            st.session_state.mm_last_ticker = ticker
                            st.session_state.mm_last_expiry = current_expiries
                    

                    # Display results
                    if st.session_state.mm_message:
                        if st.session_state.mm_data:
                            st.success(st.session_state.mm_message)
                            
                            # Process the market maker data
                            summary_data = process_market_maker_data(st.session_state.mm_data)
                            
                            if summary_data:
                                # Display summary metrics
                                st.write("### 📊 Market Maker Summary")
                                
                                # Create metric columns
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        label="Total Volume",
                                        value=f"{summary_data['total_volume']:,}"
                                    )
                                
                                with col2:
                                    call_color = st.session_state.call_color
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Call Volume</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {call_color};">{summary_data['call_volume']:,}</p>
                                        <p style="margin: 0; font-size: 14px; color: {call_color};">{summary_data['call_percentage']:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    put_color = st.session_state.put_color
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Put Volume</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {put_color};">{summary_data['put_volume']:,}</p>
                                        <p style="margin: 0; font-size: 14px; color: {put_color};">{summary_data['put_percentage']:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    if summary_data['call_volume'] > summary_data['put_volume']:
                                        bias = "Call Bias"
                                        bias_pct = summary_data['call_percentage'] - 50
                                        bias_color = "#00FF00"  # Green for call bias
                                    else:
                                        bias = "Put Bias"
                                        bias_pct = summary_data['put_percentage'] - 50
                                        bias_color = "#FF0000"  # Red for put bias
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Market Bias</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {bias_color};">{bias}</p>
                                        <p style="margin: 0; font-size: 14px; color: {bias_color};">{'+' if bias_pct > 0 else ''}{bias_pct:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.write("---")
                                
                                # Create and display charts
                                if summary_data['total_volume'] > 0:
                                    st.write("### 📈 Visual Analysis")
                                    
                                    fig_pie, fig_bar = create_market_maker_charts(summary_data)
                                    
                                    # Display charts in columns
                                    chart_col1, chart_col2 = st.columns(2)
                                    
                                    with chart_col1:
                                        st.plotly_chart(fig_pie, width='stretch')
                                    
                                    with chart_col2:
                                        st.plotly_chart(fig_bar, width='stretch')
                                
                                # Optional: Show data table in collapsible section
                                with st.expander("📋 View Raw Data Table", expanded=False):
                                    if not summary_data['raw_data'].empty:
                                        st.dataframe(summary_data['raw_data'], width='stretch')
                                    else:
                                        st.warning("No data available in the table.")
                            else:
                                st.warning("Unable to process market maker data. The data format may not be recognized.")
                                
                        else:
                            st.error(st.session_state.mm_message)
                    else:
                        # Show loading state or instructions
                        if not ticker:
                            st.info("💡 Enter a symbol above to view market maker positioning data.")
                        elif not current_expiries:
                            st.info("💡 Select expiration date(s) to view market maker positioning data.")
                        else:
                            st.info("🔄 Loading market maker data automatically...")
                        
                        # Show some help information
                        with st.expander("ℹ️ About Market Maker Data"):
                            st.write("""
                            **Market Maker Positioning Data from OCC:**
                            
                            - **Source**: Options Clearing Corporation (OCC)
                            - **Data Type**: Market maker volume and positioning
                            - **Update Frequency**: Daily (business days only)
                            - **Report Date**: Latest business day from 24 hours ago
                            - **Coverage**: All option types (Equity, Index, etc.)
                            
                            **How it works:**
                            - Data loads automatically when you enter a symbol and select expiration date(s)
                            - Supports multiple expiration dates - data is combined automatically
                            - Data refreshes automatically when you change the symbol or expiration selection
                            
                            **Data shows:**
                            - Call vs Put volume distribution
                            - Market maker activity breakdown
                            - Symbol-specific positioning data
                            
                            **Note**: Data availability depends on market maker activity and OCC reporting schedules.
                            """)

elif st.session_state.current_page == "Gamma Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Vanna Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Delta Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Charm Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Speed Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Vomma Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Color Exposure":
    with main_placeholder.container():
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, vomma, or color
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma",
                    "Color Exposure": "Color"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, width='stretch')
                
                # Display exposure explanation
                exp_title, exp_desc, exp_perspective_text, perspective = get_exposure_explanation(exposure_type)
                with st.expander(f"ℹ️ Understanding {exp_title} ({perspective} Perspective)", expanded=False):
                    st.markdown(f"**{exp_title}**")
                    st.markdown(exp_desc)
                    st.markdown(exp_perspective_text)

elif st.session_state.current_page == "Dashboard":
    with main_placeholder.container():
        # Create a single input for ticker with refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="dashboard_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_dashboard"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment("Dashboard", available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()

                def process_dashboard_date(t, d):
                    res = compute_greeks_and_charts(t, d, "dashboard", S)
                    if res[0] is None:
                        return None
                    return res[:2]

                calls, puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    process_dashboard_date
                )
                
                if calls.empty and puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                if True: # Maintain indentation
                    
                    date_suffix = f" ({len(selected_expiry_dates)} dates)" if len(selected_expiry_dates) > 1 else ""
                    
                    fig_gamma = create_exposure_bar_chart(calls, puts, "GEX", f"Gamma Exposure by Strike{date_suffix}", S)
                    fig_vanna = create_exposure_bar_chart(calls, puts, "VEX", f"Vanna Exposure by Strike{date_suffix}", S)
                    fig_delta = create_exposure_bar_chart(calls, puts, "DEX", f"Delta Exposure by Strike{date_suffix}", S)
                    fig_charm = create_exposure_bar_chart(calls, puts, "Charm", f"Charm Exposure by Strike{date_suffix}", S)
                    fig_speed = create_exposure_bar_chart(calls, puts, "Speed", f"Speed Exposure by Strike{date_suffix}", S)
                    fig_vomma = create_exposure_bar_chart(calls, puts, "Vomma", f"Vomma Exposure by Strike{date_suffix}", S)
                    fig_color = create_exposure_bar_chart(calls, puts, "Color", f"Color Exposure by Strike{date_suffix}", S)
                    
                    # Intraday price chart
                    intraday_data, current_price, vix_data = get_combined_intraday_data(ticker)
                    if intraday_data is None or current_price is None:
                        st.warning("No intraday data available for this ticker.")
                    else:
                        # Initialize plot with cleared shapes/annotations
                        fig_intraday = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_intraday.layout.shapes = []
                        fig_intraday.layout.annotations = []

                        # Add either candlestick or line trace based on selection
                        if st.session_state.intraday_chart_type == 'Candlestick':
                            if st.session_state.candlestick_type == 'Heikin Ashi':
                                # Calculate Heikin Ashi values
                                ha_data = calculate_heikin_ashi(intraday_data)
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=ha_data.index,
                                        open=ha_data['HA_Open'],
                                        high=ha_data['HA_High'],
                                        low=ha_data['HA_Low'],
                                        close=ha_data['HA_Close'],
                                        name="Price",
                                        increasing_line_color=st.session_state.call_color,
                                        decreasing_line_color=st.session_state.put_color,
                                        increasing_fillcolor=st.session_state.call_color,
                                        decreasing_fillcolor=st.session_state.put_color,
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                            elif st.session_state.candlestick_type == 'Hollow':
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing=dict(line=dict(color=st.session_state.call_color), fillcolor='rgba(0,0,0,0)'),
                                        decreasing=dict(line=dict(color=st.session_state.put_color), fillcolor='rgba(0,0,0,0)'),
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                            else:  # Filled candlesticks
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing_line_color=st.session_state.call_color,
                                        decreasing_line_color=st.session_state.put_color,
                                        increasing_fillcolor=st.session_state.call_color,
                                        decreasing_fillcolor=st.session_state.put_color,
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                        else:  # Line chart
                            fig_intraday.add_trace(
                                go.Scatter(
                                    x=intraday_data.index,
                                    y=intraday_data['Close'],
                                    name="Price",
                                    line=dict(color='gold'),
                                    showlegend=False
                                ),
                                secondary_y=False
                            )

                        # Add technical indicators if enabled
                        if st.session_state.get('show_technical_indicators') and st.session_state.get('selected_indicators'):
                            # Calculate technical indicators
                            indicators = calculate_technical_indicators(intraday_data)
                            
                            # Calculate Fibonacci levels if selected
                            fibonacci_levels = None
                            if "Fibonacci Retracements" in st.session_state.selected_indicators:
                                fibonacci_levels = calculate_fibonacci_levels(intraday_data)
                            
                            # Add indicators to chart
                            fig_intraday = add_technical_indicators_to_chart(fig_intraday, indicators, fibonacci_levels)

                        # Calculate base y-axis range from price data
                        price_min = intraday_data['Low'].min()
                        price_max = intraday_data['High'].max()
                        price_range = price_max - price_min
                        padding = price_range * 0.1  # 10% padding
                        y_min = price_min - padding
                        y_max = price_max + padding

                        # Add VIXY overlay if enabled
                        if st.session_state.show_vix_overlay and vix_data is not None and not vix_data.empty:
                            fig_intraday.add_trace(
                                go.Scatter(
                                    x=vix_data.index,
                                    y=vix_data['Close'],
                                    name='VIXY',
                                    line=dict(color=st.session_state.vix_color, width=2),
                                    opacity=0.9,
                                    showlegend=True
                                ),
                                secondary_y=True
                            )
                            
                            fig_intraday.update_yaxes(
                                title_text="VIXY", 
                                title_font=dict(color=st.session_state.vix_color),
                                tickfont=dict(color=st.session_state.vix_color),
                                secondary_y=True,
                                showgrid=False
                            )

                        elif st.session_state.show_vix_overlay:
                            st.warning("VIXY overlay enabled but no VIXY data available")

                        # Price annotation
                        if current_price is not None:
                            fig_intraday.add_annotation(
                                x=intraday_data.index[-1],
                                y=current_price,
                                xref='x',
                                yref='y',
                                xshift=27,
                                showarrow=False,
                                text=f"{current_price:,.2f}",
                                font=dict(color='yellow', size=st.session_state.chart_text_size)
                            )
                            y_min = min(y_min, current_price - padding)
                            y_max = max(y_max, current_price + padding)

                        # Process options data (Exposure levels)
                        # Apply perspective adjustment (Dealer = Short)
                        perspective = st.session_state.get('exposure_perspective', 'Customer')
                        calls_exp = calls.copy()
                        puts_exp = puts.copy()
                        calls_exp['OptionType'] = 'Call'
                        puts_exp['OptionType'] = 'Put'
                        added_strikes = set()

                        # Iterate through selected exposure types
                        for exposure_type in st.session_state.intraday_exposure_levels:
                            # Calculate strike range around current price (percentage-based)
                            strike_range = calculate_strike_range(current_price)
                            min_strike = current_price - strike_range
                            max_strike = current_price + strike_range
                            
                            # Filter options within strike range and apply perspective adjustment
                            calls_filtered = calls_exp[(calls_exp['strike'] >= min_strike) & (calls_exp['strike'] <= max_strike)].copy()
                            puts_filtered = puts_exp[(puts_exp['strike'] >= min_strike) & (puts_exp['strike'] <= max_strike)].copy()
                            
                            # Apply perspective adjustment if Dealer perspective is selected
                            if perspective == 'Dealer':
                                calls_filtered[exposure_type] = calls_filtered[exposure_type] * -1
                                puts_filtered[exposure_type] = puts_filtered[exposure_type] * -1
                            
                            # Calculate Net or Absolute Exposure
                            if exposure_type == 'GEX' and st.session_state.gex_type == 'Absolute':
                                # Absolute GEX: Sum of absolute values
                                calls_exp = calls_filtered.groupby('strike')[exposure_type].sum()
                                puts_exp = puts_filtered.groupby('strike')[exposure_type].sum()
                                net_exp = calls_exp.abs().add(puts_exp.abs(), fill_value=0)
                            elif exposure_type == 'GEX':
                                # Net GEX: Calls positive, Puts negative
                                net_exp = calls_filtered.groupby('strike')[exposure_type].sum().sub(puts_filtered.groupby('strike')[exposure_type].sum(), fill_value=0)
                            else:
                                # Other exposures: Additive (DEX, VEX, etc.)
                                net_exp = calls_filtered.groupby('strike')[exposure_type].sum().add(puts_filtered.groupby('strike')[exposure_type].sum(), fill_value=0)
                            
                            # Remove zero values and get top N by absolute value
                            net_exp = net_exp[net_exp != 0]
                            if not net_exp.empty:
                                # Create DataFrame for easier manipulation
                                exp_df = pd.DataFrame({
                                    'strike': net_exp.index,
                                    'Value': net_exp.values,
                                    'abs_Value': abs(net_exp.values)
                                })
                                
                                # Get top N by absolute value
                                level_count = st.session_state.get('intraday_level_count', 5)
                                top_levels = exp_df.nlargest(level_count, 'abs_Value')
                                top_levels['distance'] = abs(top_levels['strike'] - current_price)
                                
                                # Scale to include the nearest 5 levels (or fewer if count is small)
                                levels_to_scale = top_levels.nsmallest(min(len(top_levels), 5), 'distance')
                                max_val = top_levels['abs_Value'].max()

                                for row in top_levels.itertuples():
                                    if row.strike not in added_strikes and not pd.isna(row.Value) and row.Value != 0:
                                        # Calculate intensity based on value relative to max
                                        intensity = max(0.6, min(1.0, row.abs_Value / max_val))
                                        
                                        # Determine color based on sign (positive = call color, negative = put color)
                                        base_color = st.session_state.call_color if row.Value >= 0 else st.session_state.put_color
                                        
                                        # Convert hex to RGB
                                        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                                        
                                        # Create color with intensity
                                        color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})'
                                        
                                        # Line style: Solid for GEX, Dash for others
                                        dash_style = 'solid' if exposure_type == 'GEX' else 'dash'
                                        
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=row.strike,
                                            y1=row.strike,
                                            line=dict(
                                                color=color,
                                                width=2,
                                                dash=dash_style
                                            ),
                                            xref='x',
                                            yref='y',
                                            layer='below'
                                        )
                                        
                                        # Add text annotation positioned to the right of the chart
                                        label = f"{exposure_type} {format_large_number(row.Value)}"
                                        fig_intraday.add_annotation(
                                            x=0.92,
                                            y=row.strike,
                                            text=label,
                                            font=dict(color=color, size=st.session_state.chart_text_size),
                                            showarrow=False,
                                            xref="paper",  # Use paper coordinates for x
                                            yref="y",      # Use data coordinates for y
                                            xanchor="left"
                                        )
                                        added_strikes.add(row.strike)

                                # Include strikes in y-axis range
                                y_min = min(y_min, levels_to_scale['strike'].min() - padding)
                                y_max = max(y_max, levels_to_scale['strike'].max() + padding)

                        # Add Straddle if enabled
                        if st.session_state.show_straddle:
                            # Find ATM strike
                            atm_strike = min(calls['strike'], key=lambda x: abs(x - current_price))
                            
                            # Get Call and Put prices for ATM strike
                            atm_call = calls[calls['strike'] == atm_strike]
                            atm_put = puts[puts['strike'] == atm_strike]
                            
                            if not atm_call.empty and not atm_put.empty:
                                call_row = atm_call.iloc[0]
                                put_row = atm_put.iloc[0]
                                call_price = (call_row.get('bid', 0) + call_row.get('ask', 0)) / 2
                                put_price = (put_row.get('bid', 0) + put_row.get('ask', 0)) / 2
                                straddle_price = call_price + put_price
                                
                                upper_breakeven = atm_strike + straddle_price
                                lower_breakeven = atm_strike - straddle_price
                                
                                # Add Upper Breakeven Line
                                fig_intraday.add_shape(
                                    type='line',
                                    x0=intraday_data.index[0],
                                    x1=intraday_data.index[-1],
                                    y0=upper_breakeven,
                                    y1=upper_breakeven,
                                    line=dict(
                                        color="cyan",
                                        width=2,
                                        dash="dash"
                                    ),
                                    xref='x',
                                    yref='y',
                                    layer='below'
                                )
                                
                                fig_intraday.add_annotation(
                                    x=0.92,
                                    y=upper_breakeven,
                                    text=f"Upper BE {upper_breakeven:.2f}",
                                    font=dict(color="cyan", size=st.session_state.chart_text_size),
                                    showarrow=False,
                                    xref="paper",
                                    yref="y",
                                    xanchor="left"
                                )
                                
                                # Add Lower Breakeven Line
                                fig_intraday.add_shape(
                                    type='line',
                                    x0=intraday_data.index[0],
                                    x1=intraday_data.index[-1],
                                    y0=lower_breakeven,
                                    y1=lower_breakeven,
                                    line=dict(
                                        color="cyan",
                                        width=2,
                                        dash="dash"
                                    ),
                                    xref='x',
                                    yref='y',
                                    layer='below'
                                )

                                fig_intraday.add_annotation(
                                    x=0.92,
                                    y=lower_breakeven,
                                    text=f"Lower BE {lower_breakeven:.2f}",
                                    font=dict(color="cyan", size=st.session_state.chart_text_size),
                                    showarrow=False,
                                    xref="paper",
                                    yref="y",
                                    xanchor="left"
                                )
                                
                                # Update y-axis range to include straddle levels
                                y_min = min(y_min, lower_breakeven - padding)
                                y_max = max(y_max, upper_breakeven + padding)

                        # Add 1 SD Move if enabled
                        if st.session_state.show_sd_move:
                            # Find ATM strike
                            atm_strike = min(calls['strike'], key=lambda x: abs(x - current_price))
                            
                            # Get Call and Put prices for ATM strike
                            atm_call = calls[calls['strike'] == atm_strike]
                            atm_put = puts[puts['strike'] == atm_strike]
                            
                            if not atm_call.empty and not atm_put.empty:
                                try:
                                    # Determing Expiration Date and Time (t) first
                                    if 'extracted_expiry' in atm_call.columns:
                                        expiry_date = atm_call.iloc[0]['extracted_expiry']
                                    elif len(selected_expiry_dates) > 0:
                                        expiry_date = datetime.strptime(selected_expiry_dates[0], "%Y-%m-%d").date()
                                    else:
                                        expiry_date = datetime.now().date()
                                    
                                    t = calculate_time_to_expiration(expiry_date)
                                    t = max(t, 1e-5) # Safety

                                    # Manual IV Calculation
                                    r = st.session_state.get('risk_free_rate', 0.04)
                                    q_yield = 0

                                    call_row = atm_call.iloc[0]
                                    put_row = atm_put.iloc[0]
                                    
                                    # Calculate Mid Prices
                                    call_price = (call_row.get('bid', 0) + call_row.get('ask', 0)) / 2
                                    put_price = (put_row.get('bid', 0) + put_row.get('ask', 0)) / 2

                                    iv_call = calculate_implied_volatility(call_price, current_price, atm_strike, t, r, 'c', q_yield)
                                    iv_put = calculate_implied_volatility(put_price, current_price, atm_strike, t, r, 'p', q_yield)
                                    
                                    # Fallback if manual calculation failed
                                    if iv_call is None:
                                        iv_call = call_row.get('impliedVolatility', 0)
                                    if iv_put is None:
                                        iv_put = put_row.get('impliedVolatility', 0)
                                    
                                    if iv_call is not None and iv_put is not None and iv_call > 0 and iv_put > 0:
                                        avg_iv = (iv_call + iv_put) / 2
                                        
                                        # Calculate 1 SD Move
                                        # 1 SD = Price * IV * sqrt(t)
                                        sd_move = current_price * avg_iv * sqrt(t)
                                        
                                        upper_sd = current_price + sd_move
                                        lower_sd = current_price - sd_move
                                        
                                        # Add Upper 1 SD Line
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=upper_sd,
                                            y1=upper_sd,
                                            line=dict(
                                                color="#1E90FF", # Dodger Blue
                                                width=2,
                                                dash="dashdot"
                                            ),
                                            xref='x',
                                            yref='y',
                                            layer='below'
                                        )
                                        
                                        fig_intraday.add_annotation(
                                            x=0.92,
                                            y=upper_sd,
                                            text=f"+1 SD {upper_sd:.2f}",
                                            font=dict(color="#1E90FF", size=st.session_state.chart_text_size),
                                            showarrow=False,
                                            xref="paper",
                                            yref="y",
                                            xanchor="left"
                                        )
                                        
                                        # Add Lower 1 SD Line
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=lower_sd,
                                            y1=lower_sd,
                                            line=dict(
                                                color="#1E90FF",
                                                width=2,
                                                dash="dashdot"
                                            ),
                                            xref='x',
                                            yref='y',
                                            layer='below'
                                        )

                                        fig_intraday.add_annotation(
                                            x=0.92,
                                            y=lower_sd,
                                            text=f"-1 SD {lower_sd:.2f}",
                                            font=dict(color="#1E90FF", size=st.session_state.chart_text_size),
                                            showarrow=False,
                                            xref="paper",
                                            yref="y",
                                            xanchor="left"
                                        )
                                        
                                        # Update y-axis range
                                        y_min = min(y_min, lower_sd - padding)
                                        y_max = max(y_max, upper_sd + padding)
                                except Exception as e:
                                    print(f"Error calculating 1 SD Move: {e}")

                        # Ensure minimum range
                        if abs(y_max - y_min) < (current_price * 0.01):  # Minimum 1% range
                            center = (y_max + y_min) / 2
                            y_min = center * 0.99
                            y_max = center * 1.01

                        # Update layout
                        # Rename SPX to ^SPX for display
                        display_ticker = ticker
                        if ticker in ['SPX', '%5ESPX'] or ticker.replace('%5E', '^') == '^SPX':
                            display_ticker = '^SPX'

                        fig_intraday.update_layout(
                            title=dict(
                                text=f"Intraday Price for {display_ticker}",
                                font=dict(size=st.session_state.chart_text_size + 4)
                            ),
                            height=600,
                            hovermode='x unified',
                            margin=dict(r=10, l=50),
                            xaxis=dict(
                                autorange=True, 
                                rangeslider=dict(visible=False),
                                showgrid=False,
                                tickfont=dict(size=st.session_state.chart_text_size)
                            ),
                            yaxis=dict(
                                autorange=True,
                                fixedrange=False,
                                showgrid=False,
                                zeroline=False,
                                tickfont=dict(size=st.session_state.chart_text_size)
                            ),
                            showlegend=bool(st.session_state.get('show_technical_indicators') and st.session_state.get('selected_indicators')),  # Show legend when technical indicators are enabled
                            legend=dict(
                                x=0.01,
                                y=0.99,
                                xanchor="left",
                                yanchor="top",
                                bgcolor="rgba(0,0,0,0.5)",
                                bordercolor="rgba(255,255,255,0.2)",
                                borderwidth=1
                            )
                        )


                    # Volume ratio and other charts
                    call_volume = calls['volume'].sum()
                    put_volume = puts['volume'].sum()
                    fig_volume_ratio = create_donut_chart(call_volume, put_volume, len(selected_expiry_dates))
                    fig_max_pain = create_max_pain_chart(calls, puts, S, len(selected_expiry_dates))
                    
                    chart_options = [
                        "Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure",
                        "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Color Exposure", "Volume Ratio",
                        "Max Pain", "Delta-Adjusted Value Index", "Volume by Strike"
                    ]
                    default_charts = ["Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure"]
                    selected_charts = st.multiselect("Select charts to display:", chart_options, default=[
                        chart for chart in default_charts if chart in chart_options
                    ])

                    if 'saved_ticker' in st.session_state and st.session_state.saved_ticker:
                        current_price = get_current_price(st.session_state.saved_ticker)
                        if current_price:
                            # Filter for S&P 500 stocks only
                            gainers_df = get_screener_data("day_gainers", filter_sp500=True)
                            losers_df = get_screener_data("day_losers", filter_sp500=True)
                            
                            if not gainers_df.empty and not losers_df.empty:
                                # Get top 5 gainers and losers
                                top_gainers = gainers_df.head(5)
                                top_losers = losers_df.head(5)
                                market_text = (
                                    "<span style='color: gray; font-size: 14px;'>S&P 500 Gainers:</span> " +
                                    " ".join([f"<span style='color: {st.session_state.call_color}'>{gainer['symbol']}: +{gainer['regularMarketChangePercent']:.1f}%</span> "
                                            for _, gainer in top_gainers.iterrows()]) +
                                    " | <span style='color: gray; font-size: 14px;'>S&P 500 Losers:</span> " +
                                    " ".join([f"<span style='color: {st.session_state.put_color}'>{loser['symbol']}: {loser['regularMarketChangePercent']:.1f}%</span> "
                                            for _, loser in top_losers.iterrows()])
                                )
                                st.markdown(market_text, unsafe_allow_html=True)
                            
                            # Get additional market data
                            try:
                                # Use ^GSPC for SPX info
                                info_ticker = st.session_state.saved_ticker
                                if info_ticker == "MARKET" or info_ticker in ['^SPX', 'SPX', '%5ESPX']:
                                    info_ticker = '^GSPC'
                                    
                                stock_info = yf.Ticker(info_ticker).info
                                prev_close = stock_info.get('previousClose', 0)
                                day_high = stock_info.get('dayHigh', 0)
                                day_low = stock_info.get('dayLow', 0)
                                day_open = stock_info.get('regularMarketOpen', 0)
                                change = current_price - prev_close
                                change_percent = (change / prev_close) * 100
                                
                                # Get additional metrics
                                market_cap = stock_info.get('marketCap', 0)
                                market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                                
                                avg_volume = stock_info.get('averageVolume', 0)
                                current_volume = stock_info.get('volume', 0)
                                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                                
                                fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 0)
                                fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 0)
                                from_52_week_high = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
                                from_52_week_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100
                                
                                # Get options data if available (to calculate call-to-put ratio)
                                call_put_ratio_text = ""
                                options_volume_text = ""
                                
                                try:
                                    # Use the already fetched calls and puts dataframes which respect multi-expiry selection
                                    if not calls.empty or not puts.empty:
                                        call_volume = calls['volume'].sum()
                                        put_volume = puts['volume'].sum()
                                        call_oi = calls['openInterest'].sum()
                                        put_oi = puts['openInterest'].sum()
                                        
                                        if put_volume > 0:
                                            cp_volume_ratio = call_volume / put_volume
                                            cp_ratio_color = st.session_state.call_color if cp_volume_ratio > 1 else st.session_state.put_color
                                            options_volume_text = f"<span style='color: gray;'>Call Vol:</span> <span style='color: {st.session_state.call_color}'>{call_volume:,.0f}</span> | <span style='color: gray;'>Put Vol:</span> <span style='color: {st.session_state.put_color}'>{put_volume:,.0f}</span>"
                                            call_put_ratio_text = f" | <span style='color: gray;'>C/P Ratio:</span> <span style='color: {cp_ratio_color}'>{cp_volume_ratio:.2f}</span>"
                                        
                                        if put_oi > 0:
                                            cp_oi_ratio = call_oi / put_oi
                                            oi_ratio_color = st.session_state.call_color if cp_oi_ratio > 1 else st.session_state.put_color
                                            call_put_ratio_text += f" | <span style='color: gray;'>OI Ratio:</span> <span style='color: {oi_ratio_color}'>{cp_oi_ratio:.2f}</span>"
                                except Exception as e:
                                    print(f"Error calculating options stats: {e}")

                                # Create market data display
                                price_color = st.session_state.call_color if change >= 0 else st.session_state.put_color
                                change_symbol = '+' if change >= 0 else ''
                                
                                price_text = f"""
                                <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                                    <span style='font-size: 24px; color: {price_color}'>
                                        ${current_price:.2f} {change_symbol}{change:.2f} ({change_symbol}{change_percent:.2f}%)
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        Open: ${day_open:.2f} | High: ${day_high:.2f} | Low: ${day_low:.2f} | Prev Close: ${prev_close:.2f}
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        Market Cap: {market_cap_str} | Vol: {current_volume:,} ({volume_ratio:.2f}x avg)
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        52W Range: ${fifty_two_week_low:.2f} to ${fifty_two_week_high:.2f} ({from_52_week_low:.1f}% from low, {from_52_week_high:.1f}% from high)
                                    </span>
                                    {options_volume_text and f"<br><span style='color: gray; font-size: 14px;'>{options_volume_text}{call_put_ratio_text}</span>" or ""}
                                </div>
                                """
                                st.markdown(price_text, unsafe_allow_html=True)
                            except Exception as e:
                                st.markdown(f"#### Current Price: ${current_price:.2f}")
                                print(f"Error fetching additional market data: {e}")
                            
                            st.markdown("---")
                    # Display selected charts
                    if "Intraday Price" in selected_charts:
                        st.plotly_chart(
                            fig_intraday,
                            width='stretch',
                            key="Dashboard_intraday_chart",
                            config={
                                'modeBarButtonsToAdd': [
                                    'drawline',
                                    'drawopenpath',
                                    'drawcircle',
                                    'drawrect',
                                    'eraseshape'
                                ],
                                'displaylogo': False,
                                'scrollZoom': True,
                                'edits': {'shapePosition': True}
                            }
                        )
                    
                    supplemental_charts = []
                    for chart, fig in [
                        ("Gamma Exposure", fig_gamma), ("Delta Exposure", fig_delta),
                        ("Vanna Exposure", fig_vanna), ("Charm Exposure", fig_charm),
                        ("Speed Exposure", fig_speed), ("Vomma Exposure", fig_vomma),
                        ("Color Exposure", fig_color),
                        ("Volume Ratio", fig_volume_ratio), ("Max Pain", create_max_pain_chart(calls, puts, S, len(selected_expiry_dates))),
                        ("Delta-Adjusted Value Index", create_davi_chart(calls, puts, S, len(selected_expiry_dates))),
                        ("Volume by Strike", create_volume_by_strike_chart(calls, puts, S, len(selected_expiry_dates)))
                    ]:
                        if chart in selected_charts:
                            supplemental_charts.append(fig)
                    
                    for i in range(0, len(supplemental_charts), 2):
                        cols = st.columns(2)
                        for j, chart in enumerate(supplemental_charts[i:i+2]):
                            if chart is not None:
                                cols[j].plotly_chart(chart, width='stretch')



elif st.session_state.current_page == "Max Pain":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="max_pain_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_max_pain"):
                st.cache_data.clear()
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)  # Pass S to fetch_options_for_date
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                # Calculate and display max pain
                result = calculate_max_pain(all_calls, all_puts)
                if result is not None:
                    max_pain_strike, call_max_pain_strike, put_max_pain_strike, *_ = result
                    st.markdown(f"### Total Max Pain Strike: ${max_pain_strike:.2f}")
                    st.markdown(f"### Call Max Pain Strike: ${call_max_pain_strike:.2f}")
                    st.markdown(f"### Put Max Pain Strike: ${put_max_pain_strike:.2f}")
                    st.markdown(f"### Current Price: ${S:.2f}")
                    st.markdown(f"### Distance to Max Pain: ${abs(S - max_pain_strike):.2f}")
                    
                    # Create and display the max pain chart
                    fig = create_max_pain_chart(all_calls, all_puts, S, len(selected_expiry_dates))
                    if fig is not None:
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Could not calculate max pain point.")

elif st.session_state.current_page == "IV Surface":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="iv_skew_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_skew"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = get_ticker_object(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Use shared expiry selector
            selected_expiry_dates, _ = expiry_selector_fragment("IV Surface", available_dates)

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all IV data

                    # Calculate strike range using percentage-based setting
                    strike_range = calculate_strike_range(S)
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    for exp_date in selected_expiry_dates:
                        expiry_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        days_to_exp = (expiry_date - get_now_et().date()).days
                        calls, puts = fetch_options_for_date(ticker, exp_date, S)

                        # Filter strikes within range
                        calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                        puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

                        for strike in sorted(set(calls['strike'].unique()) | set(puts['strike'].unique())):
                            call_iv = calls[calls['strike'] == strike]['impliedVolatility'].mean()
                            put_iv = puts[puts['strike'] == strike]['impliedVolatility'].mean()
                            iv = np.nanmean([call_iv, put_iv])
                            if not np.isnan(iv):
                                all_data.append({
                                    'strike': strike,
                                    'days': days_to_exp,
                                    'iv': iv * 100  # Convert to percentage
                                })

                    if not all_data:
                        st.warning("No valid IV data available within strike range.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    # Create custom colorscale using call/put colors
                    call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    custom_colorscale = [
                        [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                        [0.5, 'rgb(255, 215, 0)'],  # Gold at center
                        [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                    ]

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot
                        fig = go.Figure()

                        # Filter data for single expiration
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Calculate center IV for coloring
                        center_iv = single_date_df['iv'].median()

                        # Create line segments with color gradient based on IV value
                        for i in range(len(single_date_df) - 1):
                            iv_val = single_date_df['iv'].iloc[i]
                            if iv_val >= center_iv:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                            
                            fig.add_trace(go.Scatter(
                                x=single_date_df['strike'].iloc[i:i+2],
                                y=single_date_df['iv'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
                            ))

                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )

                        # Update layout
                        padding = strike_range * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Implied Volatility (%)',
                            yaxis=dict(tickformat='.1f', ticksuffix='%'),
                            xaxis=dict(range=[min_strike - padding, max_strike + padding]),
                            width=800,
                            height=600
                        )

                    else:
                        # 3D Surface Plot
                        # Create meshgrid for interpolation
                        unique_strikes = np.linspace(min_strike, max_strike, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Interpolate surface
                        Z = griddata(
                            (df['strike'], df['days']),
                            df['iv'],
                            (X, Y),
                            method='linear',
                            fill_value=np.nan
                        )

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add IV surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=custom_colorscale,
                            colorbar=dict(
                                title=dict(text='IV %', side='right'),
                                tickformat='.1f',
                                ticksuffix='%'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['iv'].min(), df['iv'].max()], [df['iv'].min(), df['iv'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Implied Volatility (%)', tickformat='.1f', ticksuffix='%')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.current_page == "GEX Surface":
    with main_placeholder.container():
        # Layout for ticker input and refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input(
                "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
                value=st.session_state.get('saved_ticker', ''),
                key="gex_surface_ticker"
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄", key="refresh_button_gex"):
                st.cache_data.clear()
                st.rerun()

        # Format and save ticker
        ticker = format_ticker(user_ticker)
        if ticker != st.session_state.get('saved_ticker', ''):
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = get_ticker_object(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Use shared expiry selector
            selected_expiry_dates, _ = expiry_selector_fragment("GEX Surface", available_dates)

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all computed GEX data

                    # Calculate strike range using percentage-based setting
                    strike_range = calculate_strike_range(S)
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    # Capture context for threads
                    ctx = get_script_run_ctx() if get_script_run_ctx else None

                    def process_gex_date(date):
                        if add_script_run_ctx and ctx:
                            add_script_run_ctx(threading.current_thread(), ctx)
                        return compute_greeks_and_charts(ticker, date, "gex", S)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        future_to_date = {executor.submit(process_gex_date, date): date for date in selected_expiry_dates}
                        
                        for future in concurrent.futures.as_completed(future_to_date):
                            try:
                                result = future.result()
                                if result and result[0] is not None:
                                    calls, puts, _, t, selected_expiry, today = result
                                    
                                    days_to_exp = (selected_expiry - today).days
                                    
                                    # Filter and process data within strike range
                                    calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                                    puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
                                    
                                    # Get perspective multiplier
                                    # Customer = Long both calls and puts = positive gamma
                                    # Dealer = Short both calls and puts = negative gamma
                                    perspective = st.session_state.get('exposure_perspective', 'Customer')
                                    perspective_mult = -1 if perspective == 'Dealer' else 1
                                    
                                    for _, row in calls.iterrows():
                                        if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                            all_data.append({
                                                'strike': row['strike'],
                                                'days': days_to_exp,
                                                'gex': row['GEX'] * perspective_mult
                                            })
                                    
                                    for _, row in puts.iterrows():
                                        if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                            all_data.append({
                                                'strike': row['strike'],
                                                'days': days_to_exp,
                                                'gex': row['GEX'] * perspective_mult  # Same as calls - both positive for Customer
                                            })
                            except Exception as e:
                                print(f"Error processing date {future_to_date[future]}: {e}")

                    if not all_data:
                        st.warning("No valid GEX data available.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot for single expiration
                        fig = go.Figure()
                        
                        # Filter data for the single expiration date
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Group by strike and sum GEX values
                        grouped_gex = single_date_df.groupby('strike')['gex'].sum().reset_index()
                        
                        # Create line plot with color gradient based on GEX sign
                        for i in range(len(grouped_gex) - 1):
                            if grouped_gex['gex'].iloc[i] >= 0:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                                
                            fig.add_trace(go.Scatter(
                                x=grouped_gex['strike'].iloc[i:i+2],
                                y=grouped_gex['gex'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>GEX: %{y:,.0f}<extra></extra>'
                            ))
                        
                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )
                        
                        # Update layout with adjusted range
                        padding = (max_strike - min_strike) * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Profile - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Gamma Exposure',
                            width=800,
                            height=600,
                            xaxis=dict(range=[min_strike - padding, max_strike + padding])
                        )

                    else:
                        # 3D Surface Plot for multiple expirations
                        # Create meshgrid with adjusted strike range
                        padding = (max_strike - min_strike) * 0.05
                        unique_strikes = np.linspace(min_strike - padding, max_strike + padding, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Aggregate GEX values by strike and days
                        df_grouped = df.groupby(['strike', 'days'])['gex'].sum().reset_index()

                        # Interpolation
                        Z = griddata(
                            (df_grouped['strike'], df_grouped['days']),
                            df_grouped['gex'],
                            (X, Y),
                            method='linear',
                            fill_value=0
                        )

                        # Create custom colorscale using call/put colors
                        call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        
                        colorscale = [
                            [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                            [0.5, 'rgb(255, 215, 0)'],  # Gold at zero
                            [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                        ]

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add GEX surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=colorscale,
                            opacity=1.0,
                            colorbar=dict(
                                title=dict(text='Net GEX', side='right'),
                                tickformat=',.0f'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>Net GEX: %{z:,.0f}<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['gex'].min(), df['gex'].max()], [df['gex'].min(), df['gex'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Gamma Exposure', tickformat=',.0f')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.current_page == "Analysis":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="analysis_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_analysis"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Use ^GSPC for SPX historical data
            analysis_ticker = ticker
            if ticker == "MARKET" or ticker in ['^SPX', 'SPX', '%5ESPX']:
                analysis_ticker = '^GSPC'
                
            stock = yf.Ticker(analysis_ticker)
            # Fetch 1 year of historical data for maximum analysis period
            historical_data = stock.history(period="1y", interval="1d")
            
            if historical_data.empty:
                st.warning("No historical data available for this ticker.")
                st.stop()

            # Add key price metrics at the top
            recent_data = historical_data.tail(2)
            if len(recent_data) >= 2:
                prev_close = recent_data.iloc[-2]['Close']
                current_close = recent_data.iloc[-1]['Close']
                daily_change = (current_close - prev_close) / prev_close * 100
                daily_range = recent_data.iloc[-1]['High'] - recent_data.iloc[-1]['Low']
                daily_range_pct = daily_range / current_close * 100
                
                # Add price metrics in columns
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Current Price", f"${S:.2f}", f"{daily_change:.2f}%")
                with metrics_cols[1]:
                    st.metric("Daily Range", f"${daily_range:.2f}", f"{daily_range_pct:.2f}%")
                with metrics_cols[2]:
                    st.metric("52W High", f"${historical_data['High'].max():.2f}")
                with metrics_cols[3]:
                    st.metric("52W Low", f"${historical_data['Low'].min():.2f}")

            # Calculate indicators with proper padding
            lookback = 20  # Standard lookback period
            padding_data = pd.concat([
                historical_data['Close'].iloc[:lookback].iloc[::-1],  # Reverse first lookback periods
                historical_data['Close']
            ])
            
            # Calculate SMA and Bollinger Bands with padding
            sma_padded = padding_data.rolling(window=lookback).mean()
            std_padded = padding_data.rolling(window=lookback).std()
            
            # Trim padding and assign to historical_data
            historical_data['SMA'] = sma_padded[lookback:].values
            historical_data['Upper Band'] = historical_data['SMA'] + 2 * std_padded[lookback:].values
            historical_data['Lower Band'] = historical_data['SMA'] - 2 * std_padded[lookback:].values

            # Calculate RSI
            def calculate_rsi(data, period=14):
                # Add padding for RSI calculation
                padding = pd.concat([data.iloc[:period].iloc[::-1], data])
                delta = padding.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                # Return only the non-padded portion
                return rsi[period:].values

            historical_data['RSI'] = calculate_rsi(historical_data['Close'])
            
            # Calculate MACD
            historical_data['EMA12'] = historical_data['Close'].ewm(span=12, adjust=False).mean()
            historical_data['EMA26'] = historical_data['Close'].ewm(span=26, adjust=False).mean()
            historical_data['MACD'] = historical_data['EMA12'] - historical_data['EMA26']
            historical_data['Signal'] = historical_data['MACD'].ewm(span=9, adjust=False).mean()
            historical_data['Histogram'] = historical_data['MACD'] - historical_data['Signal']
            
            # Calculate Historical Volatility (20-day)
            historical_data['Log_Return'] = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
            historical_data['Volatility_20d'] = historical_data['Log_Return'].rolling(window=20).std() * np.sqrt(252) * 100

            # Create technical analysis chart
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    'Price vs. Simple Moving Average and Bollinger Bands',
                    'RSI'
                ),
                row_heights=[0.7, 0.3]
            )

            call_color = st.session_state.call_color
            put_color = st.session_state.put_color

            # Price and indicators with consistent colors
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Close'], 
                          name='Price', line=dict(color='gold')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['SMA'], 
                          name='SMA', line=dict(color='purple')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Upper Band'],
                          name='Upper Band', line=dict(color=call_color, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Lower Band'],
                          name='Lower Band', line=dict(color=put_color, dash='dash')),
                row=1, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=historical_data.index,
                          y=historical_data['RSI'],
                          name='RSI',
                          line=dict(color='turquoise')),
                row=2, col=1
            )

            # Add overbought/oversold lines to the RSI chart (row 2)
            fig.add_hline(y=70, line_dash="dash", line_color=call_color,
                         row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color=put_color,
                         row=2, col=1, annotation_text="Oversold")

            # Update layout
            fig.update_layout(
                template="plotly_dark",
                title=dict(
                    text=f"Technical Analysis for {ticker}",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 3)
                ),
                showlegend=True,
                height=800,  # Reduced to better fit with weekday returns
                legend=dict(
                    font=dict(size=st.session_state.chart_text_size)
                )
            )

            # Update axes for both rows
            for i in range(1, 3):
                fig.update_xaxes(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    row=i, col=1
                )
                fig.update_yaxes(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    row=i, col=1
                )

            # Set y-axis range for RSI
            fig.update_yaxes(range=[0, 100], row=2, col=1)

            # Display technical analysis chart
            st.plotly_chart(fig, width='stretch')
            
            # Create MACD chart
            macd_fig = make_subplots(rows=1, cols=1)
            
            macd_fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['MACD'], 
                          name='MACD', line=dict(color='blue')),
            )
            macd_fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Signal'], 
                          name='Signal', line=dict(color='red')),
            )
            
            # Add histogram as bar chart
            macd_fig.add_trace(
                go.Bar(x=historical_data.index, y=historical_data['Histogram'],
                      name='Histogram',
                      marker_color=historical_data['Histogram'].apply(
                          lambda x: call_color if x >= 0 else put_color
                      )),
            )
            
            macd_fig.update_layout(
                title=dict(
                    text="MACD Indicator",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 4)
                ),
                height=300,
                template="plotly_dark",
                legend=dict(
                    font=dict(size=st.session_state.chart_text_size)
                ),
                xaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                ),
                yaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                )
            )
            
            # Display MACD chart
            st.plotly_chart(macd_fig, width='stretch')
            
            # Historical volatility chart
            vol_fig = go.Figure()
            vol_fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Volatility_20d'],
                    name='20-Day HV',
                    line=dict(color='orange', width=2)
                )
            )
            
            vol_fig.update_layout(
                title=dict(
                    text="Historical Volatility (20-Day)",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 4)
                ),
                height=300,
                template="plotly_dark",
                xaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                ),
                yaxis=dict(
                    title="Volatility %",
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    ticksuffix="%"
                )
            )
            
            # Display volatility chart
            st.plotly_chart(vol_fig, width='stretch')

            # Add trend indicator section
            st.subheader("Technical Indicators Summary")
            
            # Calculate trend indicators
            current_price = historical_data['Close'].iloc[-1]
            sma20 = historical_data['SMA'].iloc[-1]
            sma50 = historical_data['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = historical_data['Close'].rolling(window=200).mean().iloc[-1]
            rsi = historical_data['RSI'].iloc[-1]
            macd = historical_data['MACD'].iloc[-1]
            signal = historical_data['Signal'].iloc[-1]
            
            # Create indicator cards in columns
            indicator_cols = st.columns(3)
            
            with indicator_cols[0]:
                st.markdown("**Price vs Moving Averages**")
                ma_indicators = [
                    f"Price vs 20 SMA: {'Bullish' if current_price > sma20 else 'Bearish'}",
                    f"Price vs 50 SMA: {'Bullish' if current_price > sma50 else 'Bearish'}",
                    f"Price vs 200 SMA: {'Bullish' if current_price > sma200 else 'Bearish'}"
                ]
                for ind in ma_indicators:
                    st.markdown(f"- {ind}")
            
            with indicator_cols[1]:
                st.markdown("**Momentum Indicators**")
                momentum_indicators = [
                    f"RSI (14): {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})",
                    f"MACD: {macd:.4f} vs Signal: {signal:.4f}",
                    f"MACD Signal: {'Bullish' if macd > signal else 'Bearish'}"
                ]
                for ind in momentum_indicators:
                    st.markdown(f"- {ind}")
            
            with indicator_cols[2]:
                st.markdown("**Volatility Indicators**")
                recent_vol = historical_data['Volatility_20d'].iloc[-1]
                avg_vol = historical_data['Volatility_20d'].mean()
                vol_indicators = [
                    f"Current HV (20d): {recent_vol:.2f}%",
                    f"Avg HV (1yr): {avg_vol:.2f}%",
                    f"Vol Trend: {'Above Average' if recent_vol > avg_vol else 'Below Average'}"
                ]
                for ind in vol_indicators:
                    st.markdown(f"- {ind}")

            # Add weekday returns analysis without extra spacing
            st.subheader("Weekday Returns Analysis")
            
            period = st.selectbox(
                "Select Analysis Period:",
                options=['1y', '6mo', '3mo', '1mo'], 
                format_func=lambda x: {
                    '1y': '1 Year',
                    '6mo': '6 Months', 
                    '3mo': '3 Months',
                    '1mo': '1 Month'
                }[x],
                key="weekday_returns_period"
            )

            weekday_returns = calculate_annualized_return(historical_data, period)
            weekday_fig = create_weekday_returns_chart(weekday_returns)
            st.plotly_chart(weekday_fig, width='stretch')

    st.stop()

elif st.session_state.current_page == "Delta-Adjusted Value Index":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="davi_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_davi"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                # The issue is here - we need to make sure the Greek values are computed
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    # Use compute_greeks_and_charts to ensure calc_delta is calculated
                    lambda t, d: compute_greeks_and_charts(t, d, "davi", S)[:2]
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                # Calculate days to expiry for each option
                try:
                    all_calls['t'] = all_calls['extracted_expiry'].apply(calculate_time_to_expiration)
                    all_puts['t'] = all_puts['extracted_expiry'].apply(calculate_time_to_expiration)
                except:
                    today = get_now_et().date()
                    all_calls['t'] = (all_calls['extracted_expiry'] - today).dt.days / 365.0
                    all_puts['t'] = (all_puts['extracted_expiry'] - today).dt.days / 365.0
                    
                # Calculate delta values if they don't exist
                if 'calc_delta' not in all_calls.columns:
                    all_calls = all_calls.copy()
                    all_puts = all_puts.copy()
                    
                    # Calculate days to expiry for each option
                    all_calls['t'] = all_calls['extracted_expiry'].apply(calculate_time_to_expiration)
                    all_puts['t'] = all_puts['extracted_expiry'].apply(calculate_time_to_expiration)
                    
                    # Define delta calculation function
                    def compute_delta(row, flag):
                        try:
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return None
                            t = row.get("t", None)
                            if t is None or t <= 0:
                                return None
                            K = row.get("strike", None)
                            if K is None:
                                return None
                            return calculate_greeks(flag, S, K, t, sigma)['delta']
                        except Exception:
                            return None
                    
                    # Calculate delta
                    all_calls['calc_delta'] = all_calls.apply(lambda row: compute_delta(row, "c"), axis=1)
                    all_puts['calc_delta'] = all_puts.apply(lambda row: compute_delta(row, "p"), axis=1)

                fig = create_davi_chart(all_calls, all_puts, S)
                st.plotly_chart(fig, width='stretch')

elif st.session_state.current_page == "Exposure Heatmap":
    with main_placeholder.container():
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="exposure_heatmap_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_heatmap"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            try:
                # Get current price
                S = get_current_price(ticker)
                if S is None:
                    st.error(f"Unable to fetch current price for {ticker}")
                    st.stop()
                
                # Get options data
                stock = get_ticker_object(ticker)
                available_dates = stock.options
                
                if not available_dates:
                    st.warning("No options data available for this ticker.")
                    st.stop()
                
                # Use shared expiry selector
                selected_expiry_dates, _ = expiry_selector_fragment("Exposure Heatmap", available_dates)
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date")
                    st.stop()
                
                # Exposure type selection
                exposure_types = {
                    "Gamma Exposure (GEX)": "GEX",
                    "Delta Exposure (DEX)": "DEX",
                    "Vanna Exposure (VEX)": "VEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma",
                    "Color Exposure": "Color"
                }
                
                # Initialize saved exposure type if not present
                if "saved_exposure_heatmap_type" not in st.session_state:
                    if "exposure_heatmap_type" in st.session_state:
                         st.session_state.saved_exposure_heatmap_type = st.session_state.exposure_heatmap_type
                    else:
                        st.session_state.saved_exposure_heatmap_type = list(exposure_types.keys())[0]

                # Determine index based on saved type
                try:
                    default_index = list(exposure_types.keys()).index(st.session_state.saved_exposure_heatmap_type)
                except ValueError:
                    default_index = 0

                def update_exposure_type():
                    val = st.session_state.get("exposure_heatmap_type")
                    if val is not None:
                        st.session_state.saved_exposure_heatmap_type = val

                col_type, col_norm = st.columns(2)
                with col_type:
                    selected_exposure_name = st.selectbox(
                        "Select Exposure Type:",
                        options=list(exposure_types.keys()),
                        index=default_index,
                        key="exposure_heatmap_type",
                        on_change=update_exposure_type
                    )

                with col_norm:
                    normalization_method = st.selectbox(
                        "Normalization Method:",
                        options=["Per Expiration", "Global"],
                        index=0,
                        key="heatmap_normalization"
                    )
                
                exposure_type = exposure_types[selected_exposure_name]
                
                # Calculate Greeks for all selected dates
                with st.spinner("Calculating exposures..."):
                    all_calls_with_greeks = []
                    all_puts_with_greeks = []
                    
                    # Capture context for threads
                    ctx = get_script_run_ctx() if get_script_run_ctx else None

                    def process_date_greeks(expiry_date):
                        if add_script_run_ctx and ctx:
                            add_script_run_ctx(threading.current_thread(), ctx)
                        return compute_greeks_and_charts(ticker, expiry_date, "exposure_heatmap", S)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        future_to_date = {executor.submit(process_date_greeks, date): date for date in selected_expiry_dates}
                        
                        for future in concurrent.futures.as_completed(future_to_date):
                            try:
                                result = future.result()
                                if result and result[0] is not None:
                                    calls_with_greeks, puts_with_greeks, _, _, _, _ = result
                                    all_calls_with_greeks.append(calls_with_greeks)
                                    all_puts_with_greeks.append(puts_with_greeks)
                            except Exception as e:
                                print(f"Error processing date {future_to_date[future]}: {e}")
                
                if not all_calls_with_greeks or not all_puts_with_greeks:
                    st.error("Unable to calculate exposures for the selected dates")
                    st.stop()
                
                # Combine all data
                combined_calls = pd.concat(all_calls_with_greeks, ignore_index=True)
                combined_puts = pd.concat(all_puts_with_greeks, ignore_index=True)
                
                # Get unique strikes and dates from actual data
                all_strikes = sorted(set(combined_calls['strike'].unique()) | set(combined_puts['strike'].unique()))
                dates_with_data = sorted(set(combined_calls['extracted_expiry'].unique()) | set(combined_puts['extracted_expiry'].unique()))
                
                # Filter strikes based on strike range
                strike_range = calculate_strike_range(S)
                min_strike = S - strike_range
                max_strike = S + strike_range
                filtered_strikes = [s for s in all_strikes if min_strike <= s <= max_strike]
                
                # Calculate current price index for the heatmap line
                s_index = np.interp(S, filtered_strikes, np.arange(len(filtered_strikes)))
                
                # Create heatmap data matrices
                call_exposure = np.zeros((len(filtered_strikes), len(dates_with_data)))
                put_exposure = np.zeros((len(filtered_strikes), len(dates_with_data)))
                
                strike_to_idx = {strike: i for i, strike in enumerate(filtered_strikes)}
                date_to_idx = {date: i for i, date in enumerate(dates_with_data)}
                
                # Fill matrices
                for _, row in combined_calls.iterrows():
                    if row['strike'] in strike_to_idx and row['extracted_expiry'] in date_to_idx:
                        strike_idx = strike_to_idx[row['strike']]
                        date_idx = date_to_idx[row['extracted_expiry']]
                        val = row[exposure_type]
                        if pd.notna(val):
                            call_exposure[strike_idx, date_idx] += val
                
                for _, row in combined_puts.iterrows():
                    if row['strike'] in strike_to_idx and row['extracted_expiry'] in date_to_idx:
                        strike_idx = strike_to_idx[row['strike']]
                        date_idx = date_to_idx[row['extracted_expiry']]
                        val = row[exposure_type]
                        if pd.notna(val):
                            put_exposure[strike_idx, date_idx] += val
                
                # Ensure no NaNs in exposure matrices
                call_exposure = np.nan_to_num(call_exposure)
                put_exposure = np.nan_to_num(put_exposure)
                
                # Apply perspective (Dealer = Short, flip the sign)
                perspective = st.session_state.get('exposure_perspective', 'Customer')
                if perspective == 'Dealer':
                    call_exposure = call_exposure * -1
                    put_exposure = put_exposure * -1
                
                # Calculate net exposure
                if exposure_type in ['GEX', 'GEX_notional']:
                    if st.session_state.get('gex_type', 'Net') == 'Net':
                        net_exposure = call_exposure - put_exposure
                    else: # Absolute
                        # Absolute GEX: Sum of absolute values
                        net_exposure = np.abs(call_exposure) + np.abs(put_exposure)
                elif exposure_type == 'DEX':
                    net_exposure = call_exposure + put_exposure
                else:
                    # For VEX, Charm, Speed, Vomma, we want to ADD them (like in create_exposure_bar_chart)
                    net_exposure = call_exposure + put_exposure
                
                # Ensure no NaNs in net exposure
                net_exposure = np.nan_to_num(net_exposure)
                
                # Normalize matrices for coloring
                call_exposure_norm = np.zeros_like(call_exposure)
                put_exposure_norm = np.zeros_like(put_exposure)
                net_exposure_norm = np.zeros_like(net_exposure)
                
                if normalization_method == "Global":
                    # Global Normalization
                    max_call = np.nanmax(call_exposure)
                    if max_call > 0:
                        call_exposure_norm = call_exposure / max_call
                        
                    max_put = np.nanmax(put_exposure)
                    if max_put > 0:
                        put_exposure_norm = put_exposure / max_put
                        
                    max_net_abs = np.nanmax(np.abs(net_exposure))
                    if max_net_abs > 0:
                        net_exposure_norm = net_exposure / max_net_abs
                else:
                    # Per Expiration Normalization (Default)
                    # Normalize Call Exposure
                    for col in range(call_exposure.shape[1]):
                        max_val = np.nanmax(call_exposure[:, col])
                        if max_val > 0:
                            call_exposure_norm[:, col] = call_exposure[:, col] / max_val
                            
                    # Normalize Put Exposure
                    for col in range(put_exposure.shape[1]):
                        max_val = np.nanmax(put_exposure[:, col])
                        if max_val > 0:
                            put_exposure_norm[:, col] = put_exposure[:, col] / max_val

                    # Normalize Net Exposure
                    for col in range(net_exposure.shape[1]):
                        max_abs = np.nanmax(np.abs(net_exposure[:, col]))
                        if max_abs > 0:
                            net_exposure_norm[:, col] = net_exposure[:, col] / max_abs
                
                # Final cleanup of normalized matrices
                call_exposure_norm = np.nan_to_num(call_exposure_norm)
                put_exposure_norm = np.nan_to_num(put_exposure_norm)
                net_exposure_norm = np.nan_to_num(net_exposure_norm)
                
                # Format dates for display
                date_labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates_with_data]
                
                # Get metric name
                metric_name = st.session_state.get('exposure_metric', 'Open Interest')
                delta_adjusted_label = " (Δ-Adjusted)" if st.session_state.get('delta_adjusted_exposures', False) and exposure_type != 'DEX' else ""
                
                # Calculate dynamic height based on number of strikes and text size
                base_height_per_row = st.session_state.chart_text_size * 2.5  # Adjust multiplier as needed
                heatmap_height = max(600, len(filtered_strikes) * base_height_per_row)
                
                # Create heatmaps
                call_color = st.session_state.call_color
                put_color = st.session_state.put_color

                # Create text arrays for heatmaps
                call_text = [[format_large_number(val) for val in row] for row in call_exposure]
                put_text = [[format_large_number(val) for val in row] for row in put_exposure]
                net_text = [[format_large_number(val) for val in row] for row in net_exposure]
                
                # Call Exposure Heatmap
                if st.session_state.show_calls:
                    fig_calls = go.Figure(data=go.Heatmap(
                        z=call_exposure_norm,
                        x=date_labels,
                        y=filtered_strikes,
                        text=call_text,
                        texttemplate='%{text}',
                        textfont=dict(size=st.session_state.chart_text_size),
                        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(call_color[1:3], 16)},{int(call_color[3:5], 16)},{int(call_color[5:7], 16)},0.1)'], [1, call_color]],
                        hoverongaps=False,
                        name="Call Exposure",
                        showscale=False,
                        hovertemplate='%{x}<br>$%{y}<extra></extra>',
                        zmin=0,
                        zmax=1
                    ))
                    
                    fig_calls.add_hline(
                        y=s_index,
                        line_dash="dash",
                        line_color="white",
                        opacity=0.7,
                        annotation_text=f"${S:.2f}",
                        annotation_position="right",
                        annotation_font=dict(size=st.session_state.chart_text_size)
                    )
                    
                    fig_calls.update_layout(
                        title=dict(
                            text=f"Call {selected_exposure_name}{delta_adjusted_label} ({metric_name})",
                            x=0,
                            xanchor='left',
                            font=dict(size=st.session_state.chart_text_size + 3)
                        ),
                        xaxis_title=dict(
                            text='Expiration Date',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis_title=dict(
                            text='Strike Price',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        template="plotly_dark",
                        height=heatmap_height,
                        xaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size - 2),
                            tickangle=-45,
                            type='category'
                        ),
                        yaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size),
                            type='category'
                        ),
                        margin=dict(r=100)
                    )
                    
                    st.plotly_chart(fig_calls, width='stretch')
                
                # Put Exposure Heatmap
                if st.session_state.show_puts:
                    fig_puts = go.Figure(data=go.Heatmap(
                        z=put_exposure_norm,
                        x=date_labels,
                        y=filtered_strikes,
                        text=put_text,
                        texttemplate='%{text}',
                        textfont=dict(size=st.session_state.chart_text_size),
                        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, f'rgba({int(put_color[1:3], 16)},{int(put_color[3:5], 16)},{int(put_color[5:7], 16)},0.1)'], [1, put_color]],
                        hoverongaps=False,
                        name="Put Exposure",
                        showscale=False,
                        hovertemplate='%{x}<br>$%{y}<extra></extra>',
                        zmin=0,
                        zmax=1
                    ))
                    
                    fig_puts.add_hline(
                        y=s_index,
                        line_dash="dash",
                        line_color="white",
                        opacity=0.7,
                        annotation_text=f"${S:.2f}",
                        annotation_position="right",
                        annotation_font=dict(size=st.session_state.chart_text_size)
                    )
                    
                    fig_puts.update_layout(
                        title=dict(
                            text=f"Put {selected_exposure_name}{delta_adjusted_label} ({metric_name})",
                            x=0,
                            xanchor='left',
                            font=dict(size=st.session_state.chart_text_size + 3)
                        ),
                        xaxis_title=dict(
                            text='Expiration Date',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis_title=dict(
                            text='Strike Price',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        template="plotly_dark",
                        height=heatmap_height,
                        xaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size - 2),
                            tickangle=-45,
                            type='category'
                        ),
                        yaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size),
                            type='category'
                        ),
                        margin=dict(r=100)
                    )
                    
                    st.plotly_chart(fig_puts, width='stretch')
                
                # Net Exposure Heatmap
                if st.session_state.show_net:
                    fig_net = go.Figure(data=go.Heatmap(
                        z=net_exposure_norm,
                        x=date_labels,
                        y=filtered_strikes,
                        text=net_text,
                        texttemplate='%{text}',
                        textfont=dict(size=st.session_state.chart_text_size),
                        colorscale=[[0, put_color], [0.5, 'black'], [1, call_color]],
                        hoverongaps=False,
                        name="Net Exposure",
                        showscale=False,
                        hovertemplate='%{x}<br>$%{y}<extra></extra>',
                        zmin=-1,
                        zmax=1
                    ))
                    
                    fig_net.add_hline(
                        y=s_index,
                        line_dash="dash",
                        line_color="white",
                        opacity=0.7,
                        annotation_text=f"${S:.2f}",
                        annotation_position="right",
                        annotation_font=dict(size=st.session_state.chart_text_size)
                    )
                    
                    fig_net.update_layout(
                        title=dict(
                            text=f"Net {selected_exposure_name}{delta_adjusted_label} ({metric_name})",
                            x=0,
                            xanchor='left',
                            font=dict(size=st.session_state.chart_text_size + 3)
                        ),
                        xaxis_title=dict(
                            text='Expiration Date',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis_title=dict(
                            text='Strike Price',
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        template="plotly_dark",
                        height=heatmap_height,
                        xaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size - 2),
                            tickangle=-45,
                            type='category'
                        ),
                        yaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size),
                            type='category'
                        ),
                        margin=dict(r=100)
                    )
                    
                    st.plotly_chart(fig_net, width='stretch')
                
                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                def format_currency_val(val):
                    s = format_large_number(val)
                    if val < 0:
                        return f"-${s[1:]}"
                    return f"${s}"

                with col1:
                    total_call = np.sum(call_exposure)
                    st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05);">
                            <p style="margin: 0; font-size: 0.9em; color: #888;">Total Call Exposure</p>
                            <p style="margin: 0; font-size: 1.5em; font-weight: bold; color: {call_color};">{format_currency_val(total_call)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_put = np.sum(put_exposure)
                    st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05);">
                            <p style="margin: 0; font-size: 0.9em; color: #888;">Total Put Exposure</p>
                            <p style="margin: 0; font-size: 1.5em; font-weight: bold; color: {put_color};">{format_currency_val(total_put)}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_net = np.sum(net_exposure)
                    net_color = call_color if total_net >= 0 else put_color
                    st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(255,255,255,0.05);">
                            <p style="margin: 0; font-size: 0.9em; color: #888;">Total Net Exposure</p>
                            <p style="margin: 0; font-size: 1.5em; font-weight: bold; color: {net_color};">{format_currency_val(total_net)}</p>
                        </div>
                    """, unsafe_allow_html=True)

                # Analysis Section
                st.markdown("### 🔍 Cross-Expiration Analysis")
                
                # Sum net exposure across all selected dates for each strike
                net_per_strike = np.sum(net_exposure, axis=1)
                
                # Find significant levels
                if len(net_per_strike) > 0:
                    # Top Positive Levels
                    top_pos_indices = np.argsort(net_per_strike)[-5:][::-1]
                    top_pos_indices = [i for i in top_pos_indices if net_per_strike[i] > 0]
                    
                    # Top Negative Levels
                    top_neg_indices = np.argsort(net_per_strike)[:5]
                    top_neg_indices = [i for i in top_neg_indices if net_per_strike[i] < 0]
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("#### Highest Positive Levels")
                        if top_pos_indices:
                            for idx in top_pos_indices:
                                strike = filtered_strikes[idx]
                                val = net_per_strike[idx]
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background-color: rgba(255,255,255,0.05); border-radius: 4px; margin-bottom: 4px;">
                                        <span style="font-weight: bold; font-size: 1.1em;">${strike:.2f}</span>
                                        <span style="color: {call_color}; font-weight: bold; font-size: 1.1em;">{format_currency_val(val)}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No significant positive levels.")
                            
                    with analysis_col2:
                        st.markdown("#### Highest Negative Levels")
                        if top_neg_indices:
                            for idx in top_neg_indices:
                                strike = filtered_strikes[idx]
                                val = net_per_strike[idx]
                                st.markdown(f"""
                                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background-color: rgba(255,255,255,0.05); border-radius: 4px; margin-bottom: 4px;">
                                        <span style="font-weight: bold; font-size: 1.1em;">${strike:.2f}</span>
                                        <span style="color: {put_color}; font-weight: bold; font-size: 1.1em;">{format_currency_val(val)}</span>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("No significant negative levels.")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif st.session_state.current_page == "Implied Probabilities":
    with main_placeholder.container():
        
        # Header
        st.title("🎲 Implied Probabilities Analysis")
        st.markdown("""
        **Analyze option-implied probabilities and expected moves based on market pricing.**
        
        This page calculates:
        - **16 Delta levels (1σ)** - Industry standard one standard deviation levels (16%/84% probability)
        - **30 Delta levels** - Common institutional trading levels (30%/70% probability)
        - **Implied move** - Expected price range based on straddle pricing
        - **Probability distribution** - Market-implied likelihood of price levels using Black-Scholes
        - **Trading ranges** - Expected breakout levels and support/resistance zones
        """)
        
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="implied_prob_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_implied_prob"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Fetch dividend yield
            try:
                if ticker == "MARKET":
                    q = 0
                else:
                    stock_info = get_ticker_object(ticker).info
                    q = stock_info.get('dividendYield', 0)
                    if q is None: q = 0
            except:
                q = 0

            stock = get_ticker_object(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.info("Please select at least one expiration date.")
                    st.stop()
                
                # For implied probabilities, we typically focus on the nearest expiry
                # But allow multiple for comparison
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, "implied_prob", S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Delta Levels", "📈 Probability Charts", "📋 Detailed Analysis"])
                
                with tab1:
                    st.subheader("Key Probability Metrics")
                    
                    # Calculate key metrics for the first (nearest) expiry
                    nearest_expiry = selected_expiry_dates[0]
                    nearest_calls = all_calls[all_calls['extracted_expiry'] == pd.to_datetime(nearest_expiry).date()]
                    nearest_puts = all_puts[all_puts['extracted_expiry'] == pd.to_datetime(nearest_expiry).date()]
                    
                    # Calculate implied move
                    implied_move_data = calculate_implied_move(S, nearest_calls, nearest_puts)
                    
                    # Calculate delta-based probability strikes (industry standard)
                    prob_16_data = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, 0.16, q)  # ~1 standard deviation
                    prob_30_data = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, 0.30, q)  # Common institutional level
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${S:.2f}")
                        if implied_move_data:
                            st.metric("Implied Move", f"${implied_move_data['implied_move_dollars']:.2f}", 
                                     f"({implied_move_data['implied_move_pct']:.1f}%)")
                    
                    with col2:
                        if prob_16_data and prob_16_data['strike_above']:
                            st.metric("16Δ Above (1σ) - 16%", f"${prob_16_data['strike_above']:.2f}", 
                                     f"Actual: {prob_16_data['prob_above']*100:.1f}% above")
                        if prob_16_data and prob_16_data['strike_below']:
                            st.metric("16Δ Below (1σ) - 84%", f"${prob_16_data['strike_below']:.2f}", 
                                     f"Actual: {prob_16_data['prob_below']*100:.1f}% below")
                    
                    with col3:
                        if prob_30_data and prob_30_data['strike_above']:
                            st.metric("30Δ Above - 30%", f"${prob_30_data['strike_above']:.2f}", 
                                     f"Actual: {prob_30_data['prob_above']*100:.1f}% above")
                        if prob_30_data and prob_30_data['strike_below']:
                            st.metric("30Δ Below - 70%", f"${prob_30_data['strike_below']:.2f}", 
                                     f"Actual: {prob_30_data['prob_below']*100:.1f}% below")
                    
                    # Expected trading range with better formatting
                    if implied_move_data:
                        st.subheader("📊 Expected Trading Range")
                        
                        range_col1, range_col2, range_col3 = st.columns(3)
                        with range_col1:
                            st.metric("Lower Range", f"${implied_move_data['lower_range']:.2f}")
                        with range_col2:
                            st.metric("Upper Range", f"${implied_move_data['upper_range']:.2f}")
                        with range_col3:
                            range_width = implied_move_data['upper_range'] - implied_move_data['lower_range']
                            range_pct = (range_width / S) * 100
                            st.metric("Range Width", f"${range_width:.2f}", f"({range_pct:.2f}%)")
                    
                    # Probability levels relative to current price with better formatting
                    st.subheader("🎯 Price Targets & Movement Required")
                    
                    # Create tabs for different delta levels (industry standard)
                    prob_tab1, prob_tab2 = st.tabs(["16Δ Levels (1σ) - 16%/84%", "30Δ Levels - 30%/70%"])
                    
                    with prob_tab1:
                        if prob_16_data and prob_16_data['strike_above'] and prob_16_data['strike_below']:
                            strike_16_above = prob_16_data['strike_above']
                            strike_16_below = prob_16_data['strike_below']
                            
                            # Ensure proper ordering
                            if strike_16_above < strike_16_below:
                                strike_16_above, strike_16_below = strike_16_below, strike_16_above
                            
                            distance_above = strike_16_above - S
                            distance_below = strike_16_below - S
                            distance_pct_above = (distance_above / S) * 100
                            distance_pct_below = (distance_below / S) * 100
                            
                            # Upper and lower bounds
                            bound_col1, bound_col2 = st.columns(2)
                            
                            with bound_col1:
                                st.markdown("**🔺 16Δ Upper Level (1σ) - 16% Above**")
                                st.metric("Strike Price", f"${strike_16_above:.2f}")
                                st.metric("Distance", f"${distance_above:.2f}", f"⬆️ {distance_pct_above:+.2f}%")
                                
                            with bound_col2:
                                st.markdown("**🔻 16Δ Lower Level (1σ) - 84% Above**")
                                st.metric("Strike Price", f"${strike_16_below:.2f}")
                                st.metric("Distance", f"${abs(distance_below):.2f}", f"⬇️ {distance_pct_below:+.2f}%")
                            
                            # Range analysis
                            st.markdown("**📏 Range Analysis**")
                            range_col1, range_col2, range_col3 = st.columns(3)
                            
                            with range_col1:
                                range_width = strike_16_above - strike_16_below
                                st.metric("Total Range", f"${range_width:.2f}")
                            
                            with range_col2:
                                range_pct = (range_width / S) * 100
                                st.metric("Range %", f"{range_pct:.2f}%")
                            
                            with range_col3:
                                # Calculate if current price is within the range
                                if strike_16_below <= S <= strike_16_above:
                                    position = "Within Range ✅"
                                elif S > strike_16_above:
                                    position = "Above Range ⬆️"
                                else:
                                    position = "Below Range ⬇️"
                                st.metric("Current Position", position)
                            
                            # Explanation
                            st.info(f"💡 **16 Delta Levels (1σ) - 16%/84%**: These represent approximate one standard deviation levels. The upper level has a 16% probability above, lower level has 84% probability above. There's roughly a 68% probability the stock will finish between ${strike_16_below:.2f} and ${strike_16_above:.2f} at expiration.")
                    
                    with prob_tab2:
                        if prob_30_data and prob_30_data['strike_above'] and prob_30_data['strike_below']:
                            strike_30_above = prob_30_data['strike_above']
                            strike_30_below = prob_30_data['strike_below']
                            
                            # Ensure proper ordering
                            if strike_30_above < strike_30_below:
                                strike_30_above, strike_30_below = strike_30_below, strike_30_above
                            
                            distance_above = strike_30_above - S
                            distance_below = strike_30_below - S
                            distance_pct_above = (distance_above / S) * 100
                            distance_pct_below = (distance_below / S) * 100
                            
                            # Upper and lower bounds
                            bound_col1, bound_col2 = st.columns(2)
                            
                            with bound_col1:
                                st.markdown("**🔺 30Δ Upper Level - 30% Above**")
                                st.metric("Strike Price", f"${strike_30_above:.2f}")
                                st.metric("Distance", f"${distance_above:.2f}", f"⬆️ {distance_pct_above:+.2f}%")
                                
                            with bound_col2:
                                st.markdown("**🔻 30Δ Lower Level - 70% Above**")
                                st.metric("Strike Price", f"${strike_30_below:.2f}")
                                st.metric("Distance", f"${abs(distance_below):.2f}", f"⬇️ {distance_pct_below:+.2f}%")
                            
                            # Range analysis
                            st.markdown("**📏 Range Analysis**")
                            range_col1, range_col2, range_col3 = st.columns(3)
                            
                            with range_col1:
                                range_width = strike_30_above - strike_30_below
                                st.metric("Total Range", f"${range_width:.2f}")
                            
                            with range_col2:
                                range_pct = (range_width / S) * 100
                                st.metric("Range %", f"{range_pct:.2f}%")
                            
                            with range_col3:
                                # Calculate if current price is within the range
                                if strike_30_below <= S <= strike_30_above:
                                    position = "Within Range ✅"
                                elif S > strike_30_above:
                                    position = "Above Range ⬆️"
                                else:
                                    position = "Below Range ⬇️"
                                st.metric("Current Position", position)
                            
                            # Explanation
                            st.info(f"💡 **30 Delta Levels - 30%/70%**: Common institutional trading levels. The upper level has a 30% probability above, lower level has 70% probability above. There's roughly a 40% probability the stock will finish between ${strike_30_below:.2f} and ${strike_30_above:.2f} at expiration.")
                
                with tab2:
                    st.subheader("Probability Levels Analysis")
                    
                    # Create a detailed table of probability levels
                    prob_levels = [0.10, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.84, 0.90]
                    prob_data = []
                    
                    for prob in prob_levels:
                        prob_info = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, prob, q)
                        if prob_info:
                            prob_data.append({
                                'Probability Level': f"{prob*100:.0f}%",
                                'Strike Above': f"${prob_info['strike_above']:.2f}" if prob_info['strike_above'] else "N/A",
                                'Strike Below': f"${prob_info['strike_below']:.2f}" if prob_info['strike_below'] else "N/A",
                                'Actual Prob Above': f"{prob_info['prob_above']*100:.1f}%" if prob_info['prob_above'] else "N/A",
                                'Actual Prob Below': f"{prob_info['prob_below']*100:.1f}%" if prob_info['prob_below'] else "N/A"
                            })
                    
                    if prob_data:
                        prob_df_display = pd.DataFrame(prob_data)
                        st.dataframe(prob_df_display, width='stretch')
                    
                    # Explanatory text
                    st.markdown("""
                    **Understanding Probability Levels:**
                    - **50% Probability**: Even odds - coin flip probability of being above/below
                    - **70% Probability**: High confidence levels for directional moves
                    - **16% Probability**: Approximately one standard deviation (84% chance of staying within range)
                    - **84% Probability**: Very high confidence levels, approximately one standard deviation
                    - **90% Probability**: Extreme confidence levels for range-bound strategies
                    
                    **Key Insight:** The wider the gap between "Strike Above" and "Strike Below" for the same probability, 
                    the higher the implied volatility and expected price movement.
                    """)
                
                with tab3:
                    st.subheader("Probability Visualization")
                    
                    # Calculate probability distribution
                    prob_df = calculate_probability_distribution(nearest_calls, nearest_puts, S, nearest_expiry, q)
                    
                    # Create comprehensive chart
                    if not prob_df.empty:
                        fig = create_implied_probabilities_chart(prob_df, S, prob_16_data, prob_30_data, implied_move_data)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("Could not calculate probability distribution.")
                
                with tab4:
                    st.subheader("Detailed Probability Analysis")
                    
                    # Show probability distribution table
                    if not prob_df.empty:
                        # Add additional calculations
                        prob_df['distance_from_current'] = abs(prob_df['strike'] - S)
                        prob_df['prob_above_pct'] = prob_df['prob_above'] * 100
                        prob_df['prob_below_pct'] = prob_df['prob_below'] * 100
                        
                        # Format for display
                        display_df = prob_df[['strike', 'prob_above_pct', 'prob_below_pct', 'distance_from_current']].copy()
                        display_df.columns = ['Strike', 'Prob Above (%)', 'Prob Below (%)', 'Distance from Current']
                        display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
                        display_df['Prob Above (%)'] = display_df['Prob Above (%)'].apply(lambda x: f"{x:.1f}%")
                        display_df['Prob Below (%)'] = display_df['Prob Below (%)'].apply(lambda x: f"{x:.1f}%")
                        display_df['Distance from Current'] = display_df['Distance from Current'].apply(lambda x: f"${x:.2f}")
                        
                        figure = go.Figure()
                        figure.add_trace(go.Bar(
                            x=display_df['Strike'],
                            y=display_df['Prob Above (%)'],
                            name="Strike Above price",
                            marker_color="#006E90",
                        ))
                        figure.add_trace(go.Bar(
                            x=display_df['Strike'],
                            y=display_df['Prob Below (%)'],
                            name="Strike Below price",
                            marker_color="#F564A9",
                        ))
                        figure.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="yellow",
                            opacity=0.7,
                            annotation_text=f"current price : {S}",
                            annotation_position="top",
                        )

                        figure.update_xaxes(range=[S-(S*0.025),S+(S*0.025)])

                        st.dataframe(display_df, width='stretch')
                        st.subheader("Detailed Probability Visualization")
                        st.plotly_chart(figure, width='stretch')
                    
                    # Additional metrics
                    if implied_move_data:
                        st.subheader("Implied Move Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**ATM Straddle Analysis:**")
                            st.write(f"ATM Strike: ${implied_move_data['atm_strike']:.2f}")
                            st.write(f"Straddle Price: ${implied_move_data['straddle_price']:.2f}")
                            st.write(f"Implied Move: {implied_move_data['implied_move_pct']:.2f}%")
                        
                        with col2:
                            st.write("**Breakeven Levels:**")
                            st.write(f"Upper Breakeven: ${implied_move_data['upper_range']:.2f}")
                            st.write(f"Lower Breakeven: ${implied_move_data['lower_range']:.2f}")
                            
                            # Calculate probability of staying within range
                            if not prob_df.empty:
                                within_range = prob_df[
                                    (prob_df['strike'] >= implied_move_data['lower_range']) & 
                                    (prob_df['strike'] <= implied_move_data['upper_range'])
                                ]
                                if not within_range.empty:
                                    prob_within = within_range['prob_above'].iloc[-1] - within_range['prob_above'].iloc[0]
                                    st.write(f"Prob within range: {prob_within*100:.1f}%")
                    
                    st.markdown("""
                    **Note:** Probabilities are derived from option delta values and implied volatilities. 
                    These represent the market's implied view of future price movements, not predictions.
                    """)

# -----------------------------------------
# Auto-refresh
# -----------------------------------------
# Check if we're on the OI & Volume page and if market maker data has been fetched
is_market_maker_active = (
    st.session_state.current_page == "OI & Volume" and
    st.session_state.get('mm_data') is not None
)

# Only auto-refresh if not on market maker tab with active data
if not is_market_maker_active:
    refresh_rate = float(st.session_state.get('refresh_rate', 10))  # Convert to float
    time.sleep(refresh_rate)
    st.rerun()
