import numpy as np

data = np.genfromtxt("freshflow_sales_5000.csv", delimiter=",",
                     skip_header=1, filling_values=np.nan, ndmin=2)

print("FRESHFLOW JUICE COMPANY")
print("Rows and columns:", data.shape)
print("Data type:", data.dtype)
print("Number of values:", data.size)

transaction_id = data[:, 0]
day = data[:, 1]
outlet = data[:, 2]
product = data[:, 3]
category = data[:, 4]
quantity = data[:, 5]
price = data[:, 6]
cost = data[:, 7]
discount = data[:, 8]
stock = data[:, 9]
payment = data[:, 10]
rating = data[:, 11]

print("\nMISSING VALUES")
columns = ["ID", "Day", "Outlet", "Product", "Category", "Quantity",
           "Price", "Cost", "Discount", "Stock", "Payment", "Rating"]

for i in range(data.shape[1]):
    print(columns[i], ":", np.isnan(data[:, i]).sum())

print("\nDATA AUDIT")

valid_id = np.isfinite(transaction_id) & (transaction_id > 0) & (transaction_id == np.floor(transaction_id))
id_values, id_counts = np.unique(transaction_id[valid_id], return_counts=True)
duplicate_ids = id_values[id_counts > 1]
duplicate_rows = np.isin(transaction_id, duplicate_ids)

print("Duplicate ID rows:", duplicate_rows.sum())

bad_day = ~np.isfinite(day) | (day < 1) | (day > 30) | (day != np.floor(day))
bad_outlet = ~np.isfinite(outlet) | ~np.isin(outlet, [1, 2, 3])
bad_product = ~np.isfinite(product) | ~np.isin(product, [1, 2, 3, 4, 5, 6, 7, 8])
bad_category = ~np.isfinite(category) | ~np.isin(category, [1, 2, 3, 4])
bad_quantity = ~np.isfinite(quantity) | (quantity <= 0) | (quantity != np.floor(quantity))
bad_price = ~np.isfinite(price) | (price <= 0)
bad_cost = ~np.isfinite(cost) | (cost < 0)
bad_discount = ~np.isfinite(discount) | (discount < 0) | (discount > 1)
bad_stock = ~np.isfinite(stock) | (stock < 0)
bad_payment = ~np.isfinite(payment) | ~np.isin(payment, [1, 2, 3, 4])

expected_category = np.zeros(len(product))
expected_category[np.isin(product, [1, 2, 3, 4])] = 1
expected_category[np.isin(product, [5, 6])] = 2
expected_category[product == 7] = 3
expected_category[product == 8] = 4

category_mismatch = (expected_category != 0) & (category != expected_category)

print("Invalid day:", bad_day.sum())
print("Invalid outlet:", bad_outlet.sum())
print("Invalid product:", bad_product.sum())
print("Invalid category:", bad_category.sum())
print("Category mismatch:", category_mismatch.sum())
print("Invalid quantity:", bad_quantity.sum())
print("Invalid price:", bad_price.sum())
print("Invalid cost:", bad_cost.sum())
print("Invalid discount:", bad_discount.sum())
print("Invalid stock:", bad_stock.sum())
print("Invalid payment:", bad_payment.sum())

bad_data = (bad_day | bad_outlet | bad_product | bad_category |
            category_mismatch | bad_quantity | bad_price | bad_cost |
            bad_discount | bad_stock | bad_payment | duplicate_rows |
            ~valid_id)

good = ~bad_data
clean_data = data[good].copy()

print("\nCLEANING SUMMARY")
print("Original rows:", len(data))
print("Rows kept:", len(clean_data))
print("Rows flagged:", bad_data.sum())
print("Raw data is unchanged. Flagged rows are not used in the main calculations.")

quantity = clean_data[:, 5]
price = clean_data[:, 6]
cost = clean_data[:, 7]
discount = clean_data[:, 8]
day = clean_data[:, 1]
outlet = clean_data[:, 2]
product = clean_data[:, 3]
rating = clean_data[:, 11]

gross_sales = quantity * price
discount_amount = gross_sales * discount
net_revenue = gross_sales - discount_amount
cost_of_goods = quantity * cost
gross_profit = net_revenue - cost_of_goods

margin = np.zeros(len(clean_data))
margin_mask = net_revenue != 0
margin[margin_mask] = gross_profit[margin_mask] / net_revenue[margin_mask] * 100

average_price = np.zeros(len(clean_data))
average_price[quantity > 0] = net_revenue[quantity > 0] / quantity[quantity > 0]

print("\nSALES SUMMARY")
print("Total units sold:", quantity.sum())
print("Gross sales:", round(gross_sales.sum(), 2))
print("Discount amount:", round(discount_amount.sum(), 2))
print("Net revenue:", round(net_revenue.sum(), 2))
print("Cost of goods:", round(cost_of_goods.sum(), 2))
print("Gross profit:", round(gross_profit.sum(), 2))
print("Overall gross margin:", round(gross_profit.sum() / net_revenue.sum() * 100, 2), "%")
print("Average transaction value:", round(net_revenue.mean(), 2))
print("Median transaction value:", round(np.median(net_revenue), 2))
print("Average selling price:", round(average_price.mean(), 2))

print("\nOUTLET REPORT")
for outlet_number in [1, 2, 3]:
    mask = outlet == outlet_number
    print("Outlet", outlet_number)
    print("Transactions:", mask.sum())
    print("Units:", quantity[mask].sum())
    print("Revenue:", round(net_revenue[mask].sum(), 2))
    print("Gross profit:", round(gross_profit[mask].sum(), 2))

print("\nPRODUCT REPORT")
product_names = ["Orange Juice", "Mango Juice", "Apple Juice",
                 "Watermelon Juice", "Banana Smoothie",
                 "Mixed Fruit Smoothie", "Lemonade", "Iced Tea"]

for product_number in range(1, 9):
    mask = product == product_number
    print(product_names[product_number - 1])
    print("Units:", quantity[mask].sum())
    print("Revenue:", round(net_revenue[mask].sum(), 2))
    print("Gross profit:", round(gross_profit[mask].sum(), 2))

print("\nDAILY SALES REPORT")
daily_revenue = np.zeros(30)

for day_number in range(1, 31):
    mask = day == day_number
    daily_revenue[day_number - 1] = net_revenue[mask].sum()
    print("Day", day_number, "Revenue:", round(daily_revenue[day_number - 1], 2))

daily_change = np.diff(daily_revenue)
print("Highest revenue day:", np.argmax(daily_revenue) + 1)
print("Lowest revenue day:", np.argmin(daily_revenue) + 1)
print("Average daily revenue:", round(daily_revenue.mean(), 2))
print("Daily revenue changes:", np.round(daily_change, 2))

print("\nDISCOUNT REPORT")
low_discount = discount < 0.10
medium_discount = (discount >= 0.10) & (discount < 0.20)
high_discount = discount >= 0.20

print("Low discount transactions:", low_discount.sum())
print("Low discount average quantity:", round(quantity[low_discount].mean(), 2))
print("Low discount average margin:", round(margin[low_discount].mean(), 2))

print("Medium discount transactions:", medium_discount.sum())
print("Medium discount average quantity:", round(quantity[medium_discount].mean(), 2))
print("Medium discount average margin:", round(margin[medium_discount].mean(), 2))

print("High discount transactions:", high_discount.sum())
print("High discount average quantity:", round(quantity[high_discount].mean(), 2))
print("High discount average margin:", round(margin[high_discount].mean(), 2))

print("\nCUSTOMER RATINGS")
valid_rating = np.isfinite(rating) & (rating >= 1) & (rating <= 5)
print("Valid ratings:", valid_rating.sum())
print("Missing or invalid ratings:", len(rating) - valid_rating.sum())

if valid_rating.sum() > 0:
    print("Average rating:", round(rating[valid_rating].mean(), 2))
    print("Median rating:", np.median(rating[valid_rating]))

print("\nSTATISTICS")
print("Revenue mean:", round(np.mean(net_revenue), 2))
print("Revenue median:", round(np.median(net_revenue), 2))
print("Revenue standard deviation:", round(np.std(net_revenue), 2))
print("Revenue minimum:", round(np.min(net_revenue), 2))
print("Revenue maximum:", round(np.max(net_revenue), 2))
print("Revenue 25th percentile:", round(np.percentile(net_revenue, 25), 2))
print("Revenue 75th percentile:", round(np.percentile(net_revenue, 75), 2))

print("\nHIGH VALUE TRANSACTIONS")
average_revenue = np.mean(net_revenue)
std_revenue = np.std(net_revenue)
high_value = net_revenue > average_revenue + (2 * std_revenue)
print("Transactions above average + 2 standard deviations:", high_value.sum())
print("Transaction IDs:", clean_data[high_value, 0])

print("\nDISCOUNT SCENARIOS")
scenario_one_discount = 0.10
scenario_two_discount = 0.15

scenario_one_revenue = gross_sales * (1 - scenario_one_discount)
scenario_two_revenue = gross_sales * (1 - scenario_two_discount)

scenario_one_profit = scenario_one_revenue - cost_of_goods
scenario_two_profit = scenario_two_revenue - cost_of_goods

print("Current revenue:", round(net_revenue.sum(), 2))
print("Revenue with 10% discount:", round(scenario_one_revenue.sum(), 2))
print("Profit with 10% discount:", round(scenario_one_profit.sum(), 2))
print("Revenue with 15% discount:", round(scenario_two_revenue.sum(), 2))
print("Profit with 15% discount:", round(scenario_two_profit.sum(), 2))
print("Scenario assumes quantity and cost do not change.")

print("\nCORRELATION")
print("Discount and quantity:", round(np.corrcoef(discount, quantity)[0, 1], 3))
print("Discount and margin:", round(np.corrcoef(discount, margin)[0, 1], 3))

print("\nPROJECT FINISHED")