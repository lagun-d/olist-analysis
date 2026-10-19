import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text
import statsmodels.api as sm

# 1. Preparing the data in SQL:
# Opening a SQL database connection:
sql_connection = sqlite3.connect("olist_data_project.db")

# Turning the CSV file of each database into its own dataframe:
# customers_df = pd.read_csv("Olist Datasets/olist_customers_dataset.csv")
# geolocation_df = pd.read_csv("Olist Datasets/olist_geolocation_dataset.csv")
# order_items_df = pd.read_csv("Olist Datasets/olist_order_items_dataset.csv")
# order_payments_df = pd.read_csv("Olist Datasets/olist_order_payments_dataset.csv")
# order_reviews_df = pd.read_csv("Olist Datasets/olist_order_reviews_dataset.csv")
# orders_df = pd.read_csv("Olist Datasets/olist_orders_dataset.csv")
# products_df = pd.read_csv("Olist Datasets/olist_products_dataset.csv")
# sellers_df = pd.read_csv("Olist Datasets/olist_sellers_dataset.csv")
# product_category_name_translation_df = pd.read_csv("Olist Datasets/product_category_name_translation.csv")

# Creating a dictionary for the loop:
datasets = {
    "orders":        "olist_orders_dataset.csv",
    "customers":     "olist_customers_dataset.csv",
    "order_items":   "olist_order_items_dataset.csv",
    "reviews":       "olist_order_reviews_dataset.csv",
    "payments":      "olist_order_payments_dataset.csv",
    "products":      "olist_products_dataset.csv",
    "sellers":       "olist_sellers_dataset.csv",
    "geolocation":   "olist_geolocation_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv"
}

# The loop, to turn the CSV file of each database into its own dataframe,
# and to then convert each into a database table:
for dataset, file_name in datasets.items():
    df = pd.read_csv(f"Olist Datasets/{file_name}")
    df.to_sql(dataset, sql_connection, if_exists="replace", index=False)

# 2. Exploring the database:
# Observing the column names of each dataset
every_column_name_list = []
for dataset in datasets:
    columns_df = pd.read_sql_query(f"PRAGMA table_info( {dataset} )", sql_connection)
    print("Column Names for " + dataset + ": ")
    column_names = list(columns_df["name"])
    index = 1
    for column_name in column_names:
        print(f"\t {index}. {column_name}")
        index += 1
        every_column_name_list.append(column_name)
    print()

# To explore the data, the relationships between the datasets can be determined
# For that, the join keys, which usually have the same names in different datasets, can be determined
# To find the join keys, only values which appear more than once in every_column_name_list can be ascertained
# However, finding these values can also give duplicate values which are not join keys, so the list will have to be further investigated
# Here is a function which only leaves duplicate values in a list:
def unique_value_remover(input_list):
    seen = set()
    duplicate_values = set()
    for value in input_list:
        if value in seen:
            duplicate_values.add(value)
        else:
            seen.add(value)
    return list(duplicate_values)

# Applying it to the column names should give the duplicate values:
duplicate_values = unique_value_remover(every_column_name_list)
duplicate_values.sort()
print(duplicate_values)

# The values in duplicate_values are ['seller_id', 'product_category_name', 'product_id', 'order_id', 'customer_id']
# All of these seem to be join keys
# Looking through the printed column names, one can also see customer_unique_id, which also seems to be a join key, albeit not in duplicate_values, as that column name is not duplicated in any other dataset
# Other column names do not seem to be join keys
# The join keys for these datasets should therefore be duplicate_values, along with customer_unique_id:
join_keys = duplicate_values + ['customer_unique_id']
join_keys.sort()
print(join_keys)

print()

# Now, looking at the first 5 rows in each dataset:
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
for dataset in datasets:
    print(f"From Dataset {dataset}:")
    print(pd.read_sql_query(f"SELECT * FROM {dataset} LIMIT 5", sql_connection).T) # Transposed for better readability
    print()

# 3. Finding Insights
# Based on the observable rows of the dataset, there are many different questions that can be answered
# There are 4 questions that I am interested in finding an answer for:
# Question 1: Does a delay in the time it takes to deliver an order correlate with its review score?
# Question 2: What is the average review score for each product category, and what revenue does each yield?
# Question 3: Does a customer's experience with an order correlate with whether they would make a repeat purchase?
# Question 4: What is the earned revenue, and how many orders do customers place, in each month of the year?
# If correlations are found, an underlying causal relationship may exist, which can be further investigated through experimentation

# 3.1: Correlation between delivery delay and review score:
# Relevant datasets: orders and reviews
# Creating and selecting a variable signifying the delivery delay, and selecting the review score variable:
query_1_df = pd.read_sql_query("""
    SELECT julianday(o.order_delivered_customer_date)
         - julianday(o.order_estimated_delivery_date) AS delay,
           r.review_score
    FROM orders o
    JOIN reviews r ON o.order_id = r.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
""", sql_connection)

# To visually demonstrate the correlation, a visualization can be used:
# Since there is an ordinal variable (review_score) and a continuous variable (delay), a box plot can be appropriate:
# Creating a dataframe of delays for each review score value:
query_1_df_to_visualize = []
for s in range(1, 6):
    query_1_df_to_visualize.append(query_1_df.loc[query_1_df["review_score"] == s, "delay"])
# Creating the boxplot:
plt.boxplot(query_1_df_to_visualize, showfliers=False, showmeans=True, orientation="horizontal")
plt.xlabel("Delivery delay (days)")
plt.ylabel("Review score")
plt.axvline(0)
# Showing the mean value of each Review score on the plot:
means = [grouping.mean() for grouping in query_1_df_to_visualize]
# Displaying the means as text:
for position, mean in enumerate(means, start=1):
    plt.text(mean, position + 0.3, f"{mean:.1f}")
# Saving the plot:
plt.savefig("3.1_delay_vs_review_score.png", bbox_inches="tight")
plt.close()

# The plot shows a clear negative correlation between the delivery delay and the review score
# This can be quantified using a regression:
result_1 = stats.linregress(query_1_df["delay"], query_1_df["review_score"])
print(f"slope: {result_1.slope} stars per day")
print(f"intercept: {result_1.intercept}, the predicted score for a delivery made on time")
print(f"r-squared: {result_1.rvalue ** 2}")
# Based on the regression results, each extra day the delivery takes is associated with a .0338 lower review score on average
# A month's delay therefore correlates with a decrease in review scores by .0338 (slope) * 30.4167 (average days in a month) = 1.02808446
# Which is a decrease in roughly 1 star for a month of extra delivery time
# An R-squared value of around .071 shows that delivery timing explains about 7% of the variation in review scores
# Therefore, most of the variation is explained by other factors, for example: product quality, customer expectations, etc.

# 3.2: Revenue and average review score by product category:
# Relevant datasets: orders, order_items, reviews, products, and product_category_name_translation
print()
query_2_df = pd.read_sql_query("""
    SELECT oi.order_id,
           oi.order_item_id,
           oi.price,
           p.product_category_name,
           t.product_category_name_english,
           r.review_score
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_category_name_translation t
           ON p.product_category_name = t.product_category_name
    LEFT JOIN reviews r ON oi.order_id = r.order_id
    WHERE o.order_status = 'delivered'
""", sql_connection)
# Only delivered orders are included, since undelivered orders are refunded and generate no revenue
# The average review scores therefore reflect satisfaction for delivered orders only

# Creating a new category variable, using the English name where it's available, otherwise using the Portuguese name, otherwise using "unknown"
query_2_df["category"] = (query_2_df["product_category_name_english"]
                          .fillna(query_2_df["product_category_name"])
                          .fillna("unknown"))

# Revenue per category with each item counted only once:
items = query_2_df.drop_duplicates(subset=["order_id", "order_item_id"])
revenue = items.groupby("category")["price"].sum()
number_of_orders = items.groupby("category")["order_id"].nunique()

# Average review per category with each order's review counted only once per category:
reviews = query_2_df.drop_duplicates(subset=["order_id", "category", "review_score"])
avg_review_score = reviews.groupby("category")["review_score"].mean()

# Combining it all into one dataframe, sorted by revenue:
query_2_summary = (pd.DataFrame({"revenue": revenue, "number_of_orders": number_of_orders,"avg_review_score": avg_review_score})
                    .sort_values("revenue", ascending=False))

print(query_2_summary)

# Since there are 2 continuous variables, a scatterplot would be best:
# Choosing only categories with at least 100 orders, so the averages are reliable:
query_2_plot_df = query_2_summary[query_2_summary["number_of_orders"] >= 100]
# Creating the scatterplot:
plt.figure(figsize=(14, 8))
plt.scatter(query_2_plot_df["revenue"], query_2_plot_df["avg_review_score"])
# Labelling the plot:
plt.xlabel("Revenue (Brazilian Real)")
plt.ylabel("Average review score")
plt.title("Revenue vs. average review score by product category")
# Since the graph seems to be skewed, it would be better to log-transform it:
plt.xscale("log")
# Labelling the categories:
texts = [plt.text(row["revenue"], row["avg_review_score"], category, fontsize=8)
         for category, row in query_2_plot_df.iterrows()]
adjust_text(texts)
# Adding a line to show the average review score:
plt.axhline(reviews["review_score"].mean())
# Saving the plot:
plt.savefig("3.2_revenue_vs_review_by_category.png", bbox_inches="tight")
plt.close()
# Through the plot, an overview of the revenue and average reviews for each category can be observed
# A regression can show the correlation between the 2 variables:
result_2 = stats.linregress(np.log10(query_2_plot_df["revenue"]), query_2_plot_df["avg_review_score"])
print(f"slope: {result_2.slope} stars for each tenfold increase in revenue")
print(f"r-squared: {result_2.rvalue**2}")
# The slope shows that the revenue generated by product categories has almost no association with review scores
# The R-squared of around 0.00005 shows that revenue explains almost none of the variation in category ratings

# 3.3: Correlation between customer's order experience and their willingness to make a repeat purchase:
# Relevant datasets: orders, customers and reviews
query_3_df = pd.read_sql_query("""
    SELECT c.customer_unique_id,
           o.order_id,
           o.order_purchase_timestamp,
           r.review_score
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN reviews r ON o.order_id = r.order_id
""", sql_connection)
# Unlike in 3.2, both delivered and undelivered orders are included, since customers can order items that do not get
# delivered in repeat purchases and because an undelivered order would be a part of a customer's order experience

# Number of orders per customer (customer_unique_id identifies the customer, customer_id identifies the order):
orders_per_customer = query_3_df.groupby("customer_unique_id")["order_id"].nunique()

# Each customer's first order:
query_3_df = query_3_df.sort_values("order_purchase_timestamp")
first_orders = query_3_df.drop_duplicates(subset="customer_unique_id", keep="first").copy()

# Flagging whether the customer ever ordered again:
first_orders["repeat_customer"] = first_orders["customer_unique_id"].map(orders_per_customer) > 1

# Customers whose first order has no review cannot be grouped by review score, so they are excluded:
reviewed_first_orders = first_orders.dropna(subset=["review_score"])

# Repeat purchase rate (%) by the review score of the first order:
repeat_rate = reviewed_first_orders.groupby("review_score")["repeat_customer"].mean() * 100
print(repeat_rate)

# Overall repeat rate (%) among customers with a reviewed first order:
overall_repeat_rate = reviewed_first_orders["repeat_customer"].mean() * 100
print(f"Overall repeat rate: {overall_repeat_rate:.2f}%")
# Only around 3.11% of customers reordered after their first order

# Plotting a bar chart to better visualize the repeat rates:
plt.bar(repeat_rate.index, repeat_rate.values)
plt.xlabel("Review score of first order")
plt.ylabel("Repeat purchase rate (%)")
plt.title("Repeat purchase rate by first-order review score")
plt.axhline(overall_repeat_rate)
plt.savefig("3.3_repeat_rate_by_review_score.png", bbox_inches="tight")
plt.close()
# Preliminarily looking at the bar chart, there seems to be no correlation between review scores and the repeat rate
# To quantify this, a regression can be done. A logistic regression seems appropriate since the dependent variable is
# binary (whether the customer reordered or not):

# The independent variable is the review score of the customer's first order
# add_constant adds the intercept term to the model, which statsmodels does not include by default:
X = sm.add_constant(reviewed_first_orders["review_score"])
# The dependent variable is whether the customer reordered:
y = reviewed_first_orders["repeat_customer"].astype(int)
# Fitting the logistic regression:
result_3 = sm.Logit(y, X).fit()

# In logistic regression, unlike in linear regression, the coefficient is on a log-odds scale
# For each additional star, the odds of a customer reordering are multiplied by the odds ratio:
odds_ratio = np.exp(result_3.params["review_score"])
print(f"odds ratio per star: {odds_ratio}")
# Pseudo R-squared is the logistic regression counterpart to R-squared
# It is on a different scale, so it is not directly comparable to the R-squared values from 3.1 and 3.2:
print(f"pseudo r-squared: {result_3.prsquared}")

# An odds ratio of 1.017 means that for every additional star in the review score, the odds of a repeat order increase by only 1.7%
# A pseudo R-squared of around 0.00006 shows that the review score explains almost none of the variation in whether customers make a repeat order
# These results confirm the observations from the bar chart, that the correlation between review score and repeat orders is negligible

# 3.4: Revenue and number of orders per month:
# Relevant datasets: orders and order_items
query_4_df = pd.read_sql_query("""
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           COUNT(DISTINCT o.order_id) AS number_of_orders,
           SUM(CASE WHEN o.order_status = 'delivered' THEN oi.price ELSE 0 END) AS revenue
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY month
    ORDER BY month
""", sql_connection)
# Instead of selecting for the delivered orders using "WHERE", a "CASE WHEN" is used here because that way orders that
# have not been delivered can be included in the order count, but don't have to count towards revenue

print(query_4_df)
# As can be seen, there is no entry for 2016-11, and the year of 2016 has very few orders. Additionally, the orders from
# September and October 2018 have no revenue, since none of them were delivered:
print(pd.read_sql_query("""
    SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
           order_status,
           COUNT(*) AS number_of_orders
    FROM orders
    WHERE strftime('%Y-%m', order_purchase_timestamp) IN ('2018-09', '2018-10')
    GROUP BY month, order_status
""", sql_connection))
# Based on this, orders shall be excluded from the months of 2016 and from September and October 2018:
query_4_df = query_4_df[(query_4_df["month"] >= "2017-01") & (query_4_df["month"] <= "2018-08")]

# A variable denoting the revenue per order can also be added for greater insight:
query_4_df["revenue_per_order"] = query_4_df["revenue"] / query_4_df["number_of_orders"]
print(query_4_df)
# Since the revenue per order stays roughly similar month by month (between 120 and 142), the growth in revenue month-to-month
# has mostly been through an increase in the number of orders rather than an increase in the revenue per order

# Graphing a line chart for the time series data:
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
ax1.plot(query_4_df["month"], query_4_df["number_of_orders"])
ax1.set_ylabel("Number of orders")
ax1.set_title("Monthly orders, revenue and revenue per order")
ax1.set_ylim(bottom=0)
ax2.plot(query_4_df["month"], query_4_df["revenue"])
ax2.set_ylabel("Revenue (Brazilian Real)")
ax2.ticklabel_format(style="plain", axis="y")
ax2.set_ylim(bottom=0)
ax3.plot(query_4_df["month"], query_4_df["revenue_per_order"])
ax3.set_ylabel("Revenue per order (Brazilian Real)")
ax3.set_ylim(bottom=0)
plt.xticks(rotation=45)
plt.savefig("3.4_monthly_orders_and_revenue.png", bbox_inches="tight")
plt.close()
# The graphs show the number of orders and revenue increasing from January 2017 to November 2017. Beyond November 2017
# both of them remain more or less constant. The revenue per order is around the same throughout the entire period. In
# November 2017, orders increased by around 63% compared to October, whereas revenue per order decreased from 140 BRL to
# 131 BRL. This could be because Black Friday occurs in November, which is when products go on discount, and more of
# them are usually bought.

# 4. Conclusions:
# From 3.1: Delivery timing has a modest correlation with customer satisfaction, which could indicate a causal relationship between the 2
# From 3.2: Customer satisfaction (measured through review scores) differs by product category, but does not correlate with category revenue.
# From 3.3: Customer satisfaction does not, however, seem to correlate with customer retention (as measured by the repeat purchase rate). Only around 3% of customers reorder in the first place, and this rate is around the same regardless of the review score of their first order.
# From 3.4: Although revenue has increased more than sevenfold across the observed time period (2017 to 2018), that increase is more attributable to increasing order quantity rather than a change in the revenue per order.
# Taken together: growth has come almost entirely from new customers, while almost no customers return, and better experiences do not appear to change that. Retention is therefore the largest untapped opportunity.
# All of the findings are correlational, and the existence of causal relationships would have to be confirmed through experimentation

# Closing the SQL database connection:
sql_connection.close()