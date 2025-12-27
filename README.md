# ⛽ Fuel Price Optimization — Machine Learning & Streamlit Application

This project implements an end-to-end **Fuel Price Optimization System** that recommends the **optimal daily fuel retail price** to maximize total profit. The solution combines:

- historical pricing and demand data  
- competitor price behavior  
- machine learning–based demand prediction  
- price simulation and optimization logic  
- business guardrails and pricing constraints  
- a user-friendly Streamlit interface  

The goal is to support **data-driven pricing decisions** instead of manual guess-based pricing.

---

## 🎯 Business Objective

A retail fuel company can set its selling price once per day. The challenge is to select a price that:

- maximizes profit  
- remains competitive in the market  
- avoids sudden or unrealistic price jumps  
- respects minimum profit margin constraints  

📌 **Daily Profit Formula**

\[
\textbf{Profit = (Price − Cost) × Predicted Volume}
\]

The system predicts volume at different price levels, simulates multiple price scenarios, and recommends the **price with the highest expected profit**.

---

## 🧩 Solution Workflow — High-Level Overview

The solution is built in three main stages:

### 1️⃣ Data Pipeline
- reads historical fuel price and volume data  
- cleans and validates records  
- creates useful engineered features  

### 2️⃣ Machine Learning Model (Demand Prediction)
- trains a Random Forest regression model  
- predicts expected fuel volume for given market conditions  

### 3️⃣ Price Optimization Engine
- simulates candidate price values for the day  
- predicts volume for each price  
- calculates expected profit  
- applies business rules  
- selects the **profit-maximizing price**

The system is integrated inside a **Streamlit app** for practical usage.

---

## 🏗️ System Architecture

