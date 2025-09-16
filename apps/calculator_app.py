import streamlit as st
import math
import numpy as np
import pandas as pd

def run():
    st.title("🧮 Advanced Calculator")
    st.markdown("---")
    
    # Calculator mode selection
    calc_mode = st.sidebar.selectbox(
        "Calculator Mode:",
        ["Basic Calculator", "Scientific Calculator", "Unit Converter", "Financial Calculator"]
    )
    
    if calc_mode == "Basic Calculator":
        basic_calculator()
    elif calc_mode == "Scientific Calculator":
        scientific_calculator()
    elif calc_mode == "Unit Converter":
        unit_converter()
    elif calc_mode == "Financial Calculator":
        financial_calculator()

def basic_calculator():
    st.subheader("🔢 Basic Calculator")
    
    # Input numbers
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num1 = st.number_input("First Number:", value=0.0, format="%.6f")
    
    with col2:
        operation = st.selectbox("Operation:", ["+", "-", "×", "÷", "^", "%"])
    
    with col3:
        num2 = st.number_input("Second Number:", value=0.0, format="%.6f")
    
    # Calculate result
    if st.button("Calculate", type="primary"):
        try:
            if operation == "+":
                result = num1 + num2
            elif operation == "-":
                result = num1 - num2
            elif operation == "×":
                result = num1 * num2
            elif operation == "÷":
                if num2 != 0:
                    result = num1 / num2
                else:
                    st.error("❌ Division by zero is not allowed!")
                    return
            elif operation == "^":
                result = num1 ** num2
            elif operation == "%":
                if num2 != 0:
                    result = num1 % num2
                else:
                    st.error("❌ Modulo by zero is not allowed!")
                    return
            
            st.success(f"✅ Result: **{result:.6f}**")
            
            # Display calculation history
            if 'calc_history' not in st.session_state:
                st.session_state.calc_history = []
            
            calculation = f"{num1} {operation} {num2} = {result:.6f}"
            st.session_state.calc_history.append(calculation)
            
            # Show last 5 calculations
            if st.session_state.calc_history:
                st.markdown("### 📝 Recent Calculations")
                for calc in st.session_state.calc_history[-5:]:
                    st.text(calc)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def scientific_calculator():
    st.subheader("🔬 Scientific Calculator")
    
    # Input number
    number = st.number_input("Enter Number:", value=0.0, format="%.6f")
    
    # Scientific functions
    st.markdown("### Trigonometric Functions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("sin"):
            result = math.sin(math.radians(number))
            st.write(f"sin({number}°) = {result:.6f}")
    
    with col2:
        if st.button("cos"):
            result = math.cos(math.radians(number))
            st.write(f"cos({number}°) = {result:.6f}")
    
    with col3:
        if st.button("tan"):
            result = math.tan(math.radians(number))
            st.write(f"tan({number}°) = {result:.6f}")
    
    st.markdown("### Logarithmic Functions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("ln"):
            if number > 0:
                result = math.log(number)
                st.write(f"ln({number}) = {result:.6f}")
            else:
                st.error("❌ Logarithm of non-positive number is undefined!")
    
    with col2:
        if st.button("log₁₀"):
            if number > 0:
                result = math.log10(number)
                st.write(f"log₁₀({number}) = {result:.6f}")
            else:
                st.error("❌ Logarithm of non-positive number is undefined!")
    
    with col3:
        if st.button("log₂"):
            if number > 0:
                result = math.log2(number)
                st.write(f"log₂({number}) = {result:.6f}")
            else:
                st.error("❌ Logarithm of non-positive number is undefined!")
    
    st.markdown("### Other Functions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("√"):
            if number >= 0:
                result = math.sqrt(number)
                st.write(f"√{number} = {result:.6f}")
            else:
                st.error("❌ Square root of negative number is undefined!")
    
    with col2:
        if st.button("x²"):
            result = number ** 2
            st.write(f"{number}² = {result:.6f}")
    
    with col3:
        if st.button("1/x"):
            if number != 0:
                result = 1 / number
                st.write(f"1/{number} = {result:.6f}")
            else:
                st.error("❌ Division by zero is not allowed!")
    
    # Constants
    st.markdown("### 🔢 Mathematical Constants")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("π"):
            st.write(f"π = {math.pi:.10f}")
    
    with col2:
        if st.button("e"):
            st.write(f"e = {math.e:.10f}")
    
    with col3:
        if st.button("φ"):
            golden_ratio = (1 + math.sqrt(5)) / 2
            st.write(f"φ = {golden_ratio:.10f}")
    
    with col4:
        if st.button("√2"):
            st.write(f"√2 = {math.sqrt(2):.10f}")

def unit_converter():
    st.subheader("🔄 Unit Converter")
    
    conversion_type = st.selectbox(
        "Select Conversion Type:",
        ["Length", "Weight", "Temperature", "Area", "Volume"]
    )
    
    if conversion_type == "Length":
        length_converter()
    elif conversion_type == "Weight":
        weight_converter()
    elif conversion_type == "Temperature":
        temperature_converter()
    elif conversion_type == "Area":
        area_converter()
    elif conversion_type == "Volume":
        volume_converter()

def length_converter():
    st.markdown("### 📏 Length Converter")
    
    value = st.number_input("Enter Value:", value=1.0, min_value=0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_unit = st.selectbox("From:", ["meters", "kilometers", "centimeters", "millimeters", "inches", "feet", "yards", "miles"])
    
    with col2:
        to_unit = st.selectbox("To:", ["meters", "kilometers", "centimeters", "millimeters", "inches", "feet", "yards", "miles"])
    
    # Conversion factors to meters
    factors = {
        "meters": 1,
        "kilometers": 1000,
        "centimeters": 0.01,
        "millimeters": 0.001,
        "inches": 0.0254,
        "feet": 0.3048,
        "yards": 0.9144,
        "miles": 1609.34
    }
    
    if st.button("Convert"):
        # Convert to meters first, then to target unit
        meters = value * factors[from_unit]
        result = meters / factors[to_unit]
        st.success(f"✅ {value} {from_unit} = **{result:.6f}** {to_unit}")

def weight_converter():
    st.markdown("### ⚖️ Weight Converter")
    
    value = st.number_input("Enter Value:", value=1.0, min_value=0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_unit = st.selectbox("From:", ["kilograms", "grams", "pounds", "ounces", "tons"])
    
    with col2:
        to_unit = st.selectbox("To:", ["kilograms", "grams", "pounds", "ounces", "tons"])
    
    # Conversion factors to kilograms
    factors = {
        "kilograms": 1,
        "grams": 0.001,
        "pounds": 0.453592,
        "ounces": 0.0283495,
        "tons": 1000
    }
    
    if st.button("Convert"):
        kg = value * factors[from_unit]
        result = kg / factors[to_unit]
        st.success(f"✅ {value} {from_unit} = **{result:.6f}** {to_unit}")

def temperature_converter():
    st.markdown("### 🌡️ Temperature Converter")
    
    value = st.number_input("Enter Temperature:", value=0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_unit = st.selectbox("From:", ["Celsius", "Fahrenheit", "Kelvin"])
    
    with col2:
        to_unit = st.selectbox("To:", ["Celsius", "Fahrenheit", "Kelvin"])
    
    if st.button("Convert"):
        # Convert to Celsius first
        if from_unit == "Fahrenheit":
            celsius = (value - 32) * 5/9
        elif from_unit == "Kelvin":
            celsius = value - 273.15
        else:
            celsius = value
        
        # Convert from Celsius to target
        if to_unit == "Fahrenheit":
            result = celsius * 9/5 + 32
        elif to_unit == "Kelvin":
            result = celsius + 273.15
        else:
            result = celsius
        
        st.success(f"✅ {value}° {from_unit} = **{result:.2f}°** {to_unit}")

def area_converter():
    st.markdown("### 📐 Area Converter")
    
    value = st.number_input("Enter Value:", value=1.0, min_value=0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_unit = st.selectbox("From:", ["square meters", "square kilometers", "square centimeters", "square feet", "square inches", "acres", "hectares"])
    
    with col2:
        to_unit = st.selectbox("To:", ["square meters", "square kilometers", "square centimeters", "square feet", "square inches", "acres", "hectares"])
    
    # Conversion factors to square meters
    factors = {
        "square meters": 1,
        "square kilometers": 1000000,
        "square centimeters": 0.0001,
        "square feet": 0.092903,
        "square inches": 0.00064516,
        "acres": 4046.86,
        "hectares": 10000
    }
    
    if st.button("Convert"):
        sq_meters = value * factors[from_unit]
        result = sq_meters / factors[to_unit]
        st.success(f"✅ {value} {from_unit} = **{result:.6f}** {to_unit}")

def volume_converter():
    st.markdown("### 🥤 Volume Converter")
    
    value = st.number_input("Enter Value:", value=1.0, min_value=0.0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        from_unit = st.selectbox("From:", ["liters", "milliliters", "gallons", "quarts", "pints", "cups", "fluid ounces"])
    
    with col2:
        to_unit = st.selectbox("To:", ["liters", "milliliters", "gallons", "quarts", "pints", "cups", "fluid ounces"])
    
    # Conversion factors to liters
    factors = {
        "liters": 1,
        "milliliters": 0.001,
        "gallons": 3.78541,
        "quarts": 0.946353,
        "pints": 0.473176,
        "cups": 0.236588,
        "fluid ounces": 0.0295735
    }
    
    if st.button("Convert"):
        liters = value * factors[from_unit]
        result = liters / factors[to_unit]
        st.success(f"✅ {value} {from_unit} = **{result:.6f}** {to_unit}")

def financial_calculator():
    st.subheader("💰 Financial Calculator")
    
    calc_type = st.selectbox(
        "Select Calculation:",
        ["Simple Interest", "Compound Interest", "Loan Payment", "Investment Growth"]
    )
    
    if calc_type == "Simple Interest":
        st.markdown("### Simple Interest Calculator")
        principal = st.number_input("Principal Amount ($):", value=1000.0, min_value=0.0)
        rate = st.number_input("Annual Interest Rate (%):", value=5.0, min_value=0.0)
        time = st.number_input("Time Period (years):", value=1.0, min_value=0.0)
        
        if st.button("Calculate Simple Interest"):
            interest = principal * (rate / 100) * time
            total = principal + interest
            st.success(f"✅ Simple Interest: **${interest:.2f}**")
            st.success(f"✅ Total Amount: **${total:.2f}**")
    
    elif calc_type == "Compound Interest":
        st.markdown("### Compound Interest Calculator")
        principal = st.number_input("Principal Amount ($):", value=1000.0, min_value=0.0)
        rate = st.number_input("Annual Interest Rate (%):", value=5.0, min_value=0.0)
        time = st.number_input("Time Period (years):", value=1.0, min_value=0.0)
        compound_freq = st.selectbox("Compounding Frequency:", ["Annually", "Semi-annually", "Quarterly", "Monthly", "Daily"])
        
        freq_map = {"Annually": 1, "Semi-annually": 2, "Quarterly": 4, "Monthly": 12, "Daily": 365}
        n = freq_map[compound_freq]
        
        if st.button("Calculate Compound Interest"):
            amount = principal * (1 + (rate/100)/n) ** (n * time)
            interest = amount - principal
            st.success(f"✅ Compound Interest: **${interest:.2f}**")
            st.success(f"✅ Total Amount: **${amount:.2f}**")
    
    elif calc_type == "Loan Payment":
        st.markdown("### Monthly Loan Payment Calculator")
        loan_amount = st.number_input("Loan Amount ($):", value=10000.0, min_value=0.0)
        annual_rate = st.number_input("Annual Interest Rate (%):", value=5.0, min_value=0.0)
        loan_term = st.number_input("Loan Term (years):", value=5.0, min_value=0.0)
        
        if st.button("Calculate Monthly Payment"):
            monthly_rate = (annual_rate / 100) / 12
            num_payments = loan_term * 12
            
            if monthly_rate > 0:
                monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = loan_amount / num_payments
            
            total_paid = monthly_payment * num_payments
            total_interest = total_paid - loan_amount
            
            st.success(f"✅ Monthly Payment: **${monthly_payment:.2f}**")
            st.success(f"✅ Total Amount Paid: **${total_paid:.2f}**")
            st.success(f"✅ Total Interest: **${total_interest:.2f}**")

if __name__ == "__main__":
    run()