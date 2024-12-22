import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def read_csv(path_in):
	return pd.read_csv(path_in)


def describe(df):
	df_numbers = df.select_dtypes(include=["number"])
	median = df_numbers.median()
	mean = df_numbers.mean()
	std = df_numbers.std()
	q1 = df_numbers.quantile(0.25)
	q3 = df_numbers.quantile(0.75)
	iqr = q3 - q1
	count = df_numbers.count()
	min_col = df_numbers.min()
	max_col = df_numbers.max()
	mode = df_numbers.mode()
	data = {
		'count': count,
		'median': median.tolist(),
		'mean': mean.tolist(),
		'std': std.tolist(),
		'25%': q1.tolist(),
		'50%': median.tolist(),
		'75%': q3.tolist(),
		'iqr': iqr.tolist(),
		'min': min_col.tolist(),
		'max': max_col.tolist(),
		'mode': mode.values[0]
	}
	df_result = pd.DataFrame(data)
	df_axis_1 = df_result.T
	df_axis_1.columns = df_numbers.columns
	print(df_numbers.columns)


# print(df_axis_1.to_csv("describe.csv"))

def missing_data(df):
	data_na = (df.isnull().sum() / len(df)) * 100
	missing_data = pd.DataFrame({ 'Ty le thieu data': data_na })
	print(missing_data)


def check_duplicates(df):
	duplicated_rows_data = df.duplicated().sum()
	print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")


# data = df.drop_duplicates()


def line_chart(df, column, title, y_label):
	# Plotting the closing price over time
	plt.plot(df.index, df[column])
	plt.title(title)
	plt.xlabel('Date')
	plt.ylabel(y_label)
	# Rotate x-axis labels
	plt.xticks(rotation=45)
	plt.show()


def candlestick_chart(df):
	fig = go.Figure(
		data=[go.Candlestick(
			x=df.index,
			open=df['Open'],
			high=df['High'],
			low=df['Low'],
			close=df['Close_Gold']
		)]
	)
	fig.update_layout(title='Biểu đồ nến vàng', xaxis_title='Date', yaxis_title='Price')
	
	# Display the figure
	fig.show()


def bar_chart(df):
	df_volume = [(i - 5000) / (20000 - 5000) for i in df['Volume']]
	# Plotting the trading volume over time with a bar chart
	plt.bar(df.index, df_volume)
	plt.title('Khối lượng giao dịch vàng theo thời gian')
	plt.xlabel('Date')
	plt.ylabel('Volume (*1000)')
	plt.grid()
	plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
	plt.show()


def histogram(df):
	df['Return'] = df['Close_Gold'].pct_change()
	
	# Drop NaN values that result from the percentage change calculation
	stock_data = df.dropna()
	
	# Plotting the histogram of stock returns
	plt.hist(stock_data['Return'], bins=30, edgecolor='black')
	plt.title('Histogram of Gold Stock Returns')
	plt.xlabel('Return')
	plt.ylabel('Frequency')
	plt.show()


def box_plot(df):
	df['Return'] = df['Close_Gold'].pct_change()
	# return = (price_current_day - prev_of_current_day) / prev_of_current_day
	# example: price on 3/3/2000 is 2000, and 2/3/2000 is 1900
	# return = (2000 - 1900) / 1900
	print(df['Return'].describe())
	# Drop NaN values resulting from the percentage change calculation
	stock_data = df.dropna()
	
	# Plotting the box plot of stock returns
	plt.figure(figsize=(8, 6))
	sns.boxplot(y=stock_data['Return'])
	plt.title('Box Plot of Stock Returns')
	plt.xlabel('Return')
	plt.show()


def regression(df):
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import Lasso
	from sklearn.model_selection import GridSearchCV
	
	from sklearn.metrics import mean_squared_error, r2_score
	X = df[['Volume', 'SP500', 'Close_Oil', 'DollarIndex']]
	y = df['Close_Gold']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	# Sử dụng GridSearchCV để tìm giá trị alpha tối ưu
	lasso = Lasso()
	params = { 'alpha': [0.1, 0.5, 1.0, 5.0, 10.0] }
	grid_search = GridSearchCV(lasso, param_grid=params, cv=5)
	grid_search.fit(X_train, y_train)
	
	best_alpha = grid_search.best_params_['alpha']
	print("Best alpha:", best_alpha)
	lasso_model = Lasso(alpha=best_alpha)
	lasso_model.fit(X_train, y_train)
	y_pred = lasso_model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	
	print("Mean Squared Error:", mse)
	print("R2 Score:", r2)
	print("y_pred", y_pred)
	feature_importance = pd.Series(lasso_model.coef_, index=X.columns)
	print("Feature importance:\n", feature_importance)


if __name__ == '__main__':
	df = pd.read_csv('Data_gold_oil_dollar_sp500.csv')
	df["Date"] = pd.to_datetime(df["Date"])  # Chuyển đổi Date sang kiểu datetime
	df.set_index("Date", inplace=True)  # Đặt Date làm index
	# Gọi các hàm và hiển thị dữ liệu
	missing_data(df)
	check_duplicates(df)
	describe(df)
	line_chart(df, "Close_Gold", 'Giá vàng', 'Giá')
	line_chart(df, "DollarIndex", 'Chỉ số DXY', 'Điểm')
	line_chart(df, "SP500", 'Chỉ số SP500', 'Điểm')
	line_chart(df, "Close_Oil", 'Giá dầu', 'Giá')
	box_plot(df)
	bar_chart(df)
	histogram(df)
	candlestick_chart(df)