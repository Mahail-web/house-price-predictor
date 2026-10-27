"""
Utility functions for the House Price Predictor application.
Provides formatting, styling helpers, and input validation utilities.
"""

from typing import Optional, Union
import streamlit as st


def format_currency(
    value: Union[int, float],
    currency_symbol: str = "$",
    decimals: int = 0
) -> str:
    """
    Format a numeric value as a currency string with thousands separators.
    
    Parameters:
    -----------
    value : Union[int, float]
        Numeric value to format.
    currency_symbol : str
        Currency symbol to prepend (default: "$").
    decimals : int
        Number of decimal places (default: 0).
        
    Returns:
    --------
    str : Formatted currency string.
    """
    if value is None:
        return "N/A"
    try:
        if decimals > 0:
            formatted = f"{value:,.{decimals}f}"
        else:
            formatted = f"{int(round(value)):,}"
        return f"{currency_symbol}{formatted}"
    except (ValueError, TypeError):
        return f"{currency_symbol}{value}"


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    badge_color: str = "#0D9488"
) -> str:
    """
    Generate an HTML string for a polished, real-estate styled metric card.
    """
    subtext_html = f"<div style='font-size: 0.85rem; color: #64748B; margin-top: 4px;'>{subtext}</div>" if subtext else ""
    card_html = f"""
    <div style="
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border-left: 4px solid {badge_color};
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #475569;">
            {label}
        </div>
        <div style="font-size: 1.65rem; font-weight: 700; color: #0F172A; margin-top: 6px; letter-spacing: -0.02em;">
            {value}
        </div>
        {subtext_html}
    </div>
    """
    return card_html


def get_custom_css() -> str:
    """
    Return custom CSS for a real-estate inspired, sleek user interface.
    Dark navy/slate primary tones (#0F172A, #1E293B) with crisp white containers
    and subtle gold (#D97706) / teal (#0D9488) accents.
    """
    return """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Main content container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #F1F5F9 !important;
    }

    /* Custom Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 32px 36px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.1);
    }
    .header-banner h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .header-banner p {
        color: #94A3B8 !important;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .header-badge {
        display: inline-block;
        background: rgba(217, 119, 6, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }

    /* Sleek Cards */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 18px;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 28px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.3);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0F766E 0%, #115E59 100%);
        box-shadow: 0 6px 12px -2px rgba(13, 148, 136, 0.4);
        transform: translateY(-1px);
    }

    /* Input fields */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 8px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        border-bottom-color: #0D9488 !important;
    }

    /* Callouts */
    .info-callout {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 8px;
        padding: 14px 18px;
        color: #166534;
        margin-bottom: 16px;
    }
    .warning-callout {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        border-radius: 8px;
        padding: 14px 18px;
        color: #92400E;
        margin-bottom: 16px;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """
