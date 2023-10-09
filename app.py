# import libraries
import os
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask import (Flask,
                   render_template, 
                   request,
                   send_from_directory,
                   jsonify,
                   Response)
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.ticker import MultipleLocator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from plotly.tools import mpl_to_plotly


# Create Flask app
app = Flask(__name__)

# Set password
PASSWORD = "seamTA07"

# Set up basic authentication
@app.before_request
def require_basic_auth():
    auth = request.authorization
    if not auth or auth.password != PASSWORD:
        return Response(
            "Unauthorized", 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'}
        )

# Azure Database for MySQL connection string
DATABASE_CONFIG = {
    'host': 'seam-server.mysql.database.azure.com',
    'user': 'ainspireestate',
    'password': 'seamTA07',
    'database': 'housing'
}

# MySQL connection string
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqlconnector://ainspireestate:seamTA07@seam-server.mysql.database.azure.com:3306/housing"
db = SQLAlchemy(app)

# SQLAlchemy ORM definition for Melbourne Housing Data
class MelbourneHousingData(db.Model):
    __tablename__ = "melbourne_housing_data_new"
    my_row_id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String)
    suburb = db.Column(db.String)
    rooms = db.Column(db.Integer)
    bathroom = db.Column(db.Integer)
    price = db.Column(db.Double)
    latitude = db.Column(db.Double)
    longitude = db.Column(db.Double)
    car = db.Column(db.Integer)
    type = db.Column(db.String)
    distance = db.Column(db.Double)
    postcode = db.Column(db.Integer)
    landarea = db.Column(db.Double)
    council = db.Column(db.String)
    region = db.Column(db.String)
    state = db.Column(db.String)
    date = db.Column(db.String)

# index page
@app.route('/', methods=['GET'])
def index():
   print('Request for index page received')
   return render_template('index.html')

# favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')

# browsing & query properties page
@app.route('/search', methods=['GET', 'POST'])
def search():
   print('Request for search page received')
   
   # Get all suburbs from database
   suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
   
   # Get latitude and longitude of properties from the database
   coordinates = db.session.query(MelbourneHousingData.latitude, MelbourneHousingData.longitude).all()
   return render_template('search.html', 
                          suburbs=suburbs,
                           coordinates=coordinates)

@app.route('/get_types', methods=['GET'])
def get_types():
    # Get the selected suburb from the request parameters
    selectedSuburb = request.args.get('suburb')

    # query available property types for the selected suburb
    types = db.session.query(MelbourneHousingData.type).filter(MelbourneHousingData.suburb == selectedSuburb).distinct().all()
    return jsonify([type[0] for type in types])

@app.route('/get_bedrooms', methods=['GET'])
def get_bedrooms():
    # Get the selected suburb and type from the request parameters
    selectedSuburb = request.args.get('suburb')
    selectedType = request.args.get('type')

    # query available bedrooms for the selected suburb and type
    bedrooms = db.session.query(MelbourneHousingData.rooms).filter(MelbourneHousingData.suburb == selectedSuburb, MelbourneHousingData.type == selectedType).distinct().all()
    return jsonify([bedroom[0] for bedroom in bedrooms])

@app.route('/get_bathrooms', methods=['GET'])
def get_bathrooms():
    # Get the selected suburb and type from the request parameters
    selectedSuburb = request.args.get('suburb')
    selectedType = request.args.get('type')
    selectedBedrooms = request.args.get('bedroom')

    # query available bathrooms for the selected suburb, type and bedrooms
    bathrooms = db.session.query(MelbourneHousingData.bathroom).filter(MelbourneHousingData.suburb == selectedSuburb, MelbourneHousingData.type == selectedType, MelbourneHousingData.rooms == selectedBedrooms).distinct().all()
    return jsonify([bathroom[0] for bathroom in bathrooms])

@app.route('/get_carplace', methods=['GET'])
def get_carplace():
    # Get the selected suburb and type from the request parameters
    selectedSuburb = request.args.get('suburb')
    selectedType = request.args.get('type')
    selectedBedrooms = request.args.get('bedroom')
    selectedBathrooms = request.args.get('bathroom')

    # query available garages for the selected suburb, type, bedrooms and bathrooms
    garages = db.session.query(MelbourneHousingData.car).filter(MelbourneHousingData.suburb == selectedSuburb, MelbourneHousingData.type == selectedType, MelbourneHousingData.rooms == selectedBedrooms, MelbourneHousingData.bathroom == selectedBathrooms).distinct().all()
    return jsonify([garage[0] for garage in garages])

@app.route('/show_result', methods=['GET', 'POST'])
def show_result():
    properties_info = request.args.get('properties_info')
    print(properties_info)
    return render_template('search_result.html', properties_info=properties_info)

@app.route('/demand', methods=['GET', 'POST'])
def demand():
    print('Request for demand page received')
    selected_suburb = request.args.get('selected_suburb')
    suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
    return render_template('demand.html', selected_suburb=selected_suburb, suburbs=suburbs)


# query properties
@app.route('/search_property', methods=['POST'])
def search_property():
    # accept the request data from the client
    data = request.json
    suburb = data['suburb']
    property_type = data['type']
    number_of_bedrooms = data['bedrooms']
    number_of_bathrooms = data['bathrooms']
    number_of_car_places = data['carplaces']

    # search the database for properties that match the search criteria
    properties = db.session.query(MelbourneHousingData).filter(MelbourneHousingData.suburb == suburb, 
                                                                MelbourneHousingData.type == property_type,
                                                                MelbourneHousingData.rooms == number_of_bedrooms,
                                                                MelbourneHousingData.bathroom == number_of_bathrooms,
                                                                MelbourneHousingData.car == number_of_car_places).all()

    # get the coordinates of the properties
    properties_info = [(property.latitude, property.longitude, property.type, property.rooms, property.bathroom, property.car, property.price, property.suburb) for property in properties]
    
    return jsonify(properties_info)

@app.route('/suggestion', methods=['GET'])
def suggestion():
    print('Request for suggestion page received')

    suburb = request.args.get('suburb')
    property_type = request.args.get('property_type')
    bedroom = request.args.get('bedroom')
    bathroom = request.args.get('bathroom')
    carpark = request.args.get('carpark')

    # Pass the suburb, property type, bedrooms, bathrooms and car spaces to the suggestion page
    return render_template('suggestion.html', suburb=suburb, property_type=property_type, bedroom=bedroom, bathroom=bathroom, carpark=carpark)


@app.route('/get_suggestion', methods=['POST'])
def get_suggestion():
    print('Request for get suggestion received')
    # accept the request data from the client
    data = request.json
    suburb = data['suburb']
    property_type = data['property_type']
    bedrooms = data['bedroom']
    bathrooms = data['bathroom']
    car_spaces = data['carpark']

    # print("testing: ", suburb, property_type, bedrooms, bathrooms, car_spaces)

    # search the database for properties that match the search criteria
    properties = db.session.query(MelbourneHousingData).filter(MelbourneHousingData.suburb == suburb, 
                                                                MelbourneHousingData.type == property_type,
                                                                MelbourneHousingData.rooms >= bedrooms,
                                                                MelbourneHousingData.bathroom >= bathrooms,
                                                                MelbourneHousingData.car >= car_spaces).all()

    # get the coordinates of the properties
    properties_info = [(property.latitude, property.longitude, property.suburb, property.rooms, property.type, property.price, property.distance, property.postcode, property.bathroom, property.car, property.landarea) for property in properties]
    print("lenth: ", len(properties_info))

    df = pd.DataFrame(properties_info, columns=['Latitude', 'Longitude', 'Suburb', 'Rooms', 'Type', 'Price', 'Distance', 'Postcode', 'Bathroom', 'Car', 'LandArea'])

    X = df.drop('Price', axis=1)
    y = df['Price']
    postcode = X[X['Suburb'] == suburb]['Postcode'].values[0]
    new_data = pd.DataFrame({
        'Suburb': [suburb],
        'Rooms': [bedrooms],
        'Type': [property_type],
        'Bathroom': [bathrooms],
        'Car': [car_spaces],
        'Postcode': [postcode]
    })
    X = X[X['Suburb'] == new_data['Suburb'].values[0]]
    suburb_data = df[df['Suburb'] == new_data['Suburb'].values[0]]
    X_encoded = pd.get_dummies(X, columns=['Suburb', 'Type'], drop_first=True)
    weights = {
        'Rooms': 1.0,
        'Bathroom': 1.0,
        'Car': 1.0,
        'Postcode': 1.0,
        'Distance': 0.0,
        'LandArea': 0.0
    }
    for feature, weight in weights.items():
        X_encoded[feature] *= weight
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    from sklearn.neighbors import NearestNeighbors
    
    k = min(3, len(properties_info))
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(X_scaled)
    # Preprocessing the new data in the same way as the training data
    new_data_encoded = pd.get_dummies(new_data, columns=['Suburb', 'Type'], drop_first=True)

    # Ensure the encoded new data has the same columns as the training data
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Now you can scale
    new_data_scaled = scaler.transform(new_data_encoded)
    distances, indices = knn.kneighbors(new_data_scaled)
    X.iloc[indices[0]]
    exact_matches = []
    for i in range(len(distances[0])):
        if distances[0][i] == 0.0:
            exact_matches.append(indices[0][i])
    if exact_matches == []:
        # return(suburb_data.iloc[indices[0]])
        return jsonify(suburb_data.sort_values(by='Price', ascending=True).values.tolist())
        # return template with data
        # return render_template('suggestion.html', properties=suburb_data.iloc.sort_values(by='Price', ascending=True).values.tolist())
    else:
        # return(suburb_data.iloc[exact_matches].sort_values(by='Price', ascending=True))
        return jsonify(suburb_data.iloc[exact_matches].sort_values(by='Price', ascending=True).values.tolist())
        # return render_template('suggestion.html', properties=suburb_data.iloc[exact_matches].sort_values(by='Price', ascending=True).values.tolist())
    

@app.route('/avg', methods=['POST'])
def avg():
   data = request.json
   suburb = data['selected_suburb']
   avg_bedrooms = db.session.query(db.func.avg(MelbourneHousingData.rooms)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_bathrooms = db.session.query(db.func.avg(MelbourneHousingData.bathroom)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_price = db.session.query(db.func.avg(MelbourneHousingData.price)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_car = db.session.query(db.func.avg(MelbourneHousingData.car)).filter(MelbourneHousingData.suburb == suburb).scalar()
   return jsonify(round(avg_price))

@app.route('/sale', methods=['GET', 'POST'])
def sale():
    print('Request for sale page received')
    # update_linear_regression_model()
    suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
    return render_template('sale.html', suburbs=suburbs)

@app.route('/predict_price', methods=['POST'])
def predict_price():

    # Get the request data from the client
    data = request.json
    suburb = data['suburb']
    bedroom = data['bedroom']
    property_type = data['property_type']
    bathroom = data['bathroom']
    car_place = data['car_place']

    # Creating a new data
    new_data = pd.DataFrame({
        'suburb': [suburb],
        'rooms': [bedroom],
        'type': [property_type],
        'distance': [0],
        'bathroom': [bathroom],
        'car': [car_place],
        'landarea': [0]
    })

    # load the model
    rf_model = joblib.load('ai_model/random_forest_model.pkl')

    # load the train columns
    with open('ai_model/train_columns.pkl', 'rb') as file:
        train_columns = pickle.load(file)

    # Preprocessing the new data in the same way as the training data
    scaler = joblib.load('ai_model/scaler.bin')
    new_data = pd.get_dummies(new_data, columns=['suburb', 'type'], drop_first=True).reindex(columns=train_columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data)

    # Making predictions
    predicted_price = rf_model.predict(new_data_scaled)[0]

    print(f"Expected Price: {round(predicted_price)}")
    data = []
    data.append(round(predicted_price))
    selected_features = ['rooms', 'bathroom', 'var', 'distance', 'landarea']
    # Get feature importances
    importances = rf_model.feature_importances_

    # Map importances to feature names
    feature_importances = {feature: importance for feature, importance in zip(train_columns, importances) if feature in selected_features}
    
    # Display feature importances
    feature_import_dict = {}
    for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance}")
        feature_import_dict[feature] = importance
    data.append(feature_import_dict)
    return jsonify(data)

@app.route('/price_estimate', methods=['GET', 'POST'])
def price_estimate():
    price = request.args.get('price')
    suburb = request.args.get('suburb')
    return render_template('price_estimate.html', price=price, suburb=suburb)

@app.route('/intervention', methods=['GET'])
def intervention():
    print('Request for intervention page received')
    return render_template('intervention.html')


# update linear regression model in the database
@app.route('/update_linear_regression_model', methods=['GET'])
def update_linear_regression_model():
    # Import joblib
    import joblib
    # Execute a query to get data from the table
    query_result = db.session.query(MelbourneHousingData).all()

    # Convert the query result to a dataframe
    df = pd.DataFrame([(row.suburb, 
                        row.rooms,
                        row.type, 
                        row.price,
                        row.distance,
                        row.bathroom,
                        row.car,
                        row.landarea
                        ) for row in query_result], 
                        columns=['suburb',
                                 'rooms',
                                 'type', 
                                 'price',
                                 'distance',
                                 'bathroom',
                                 'car',
                                 'landarea',
                                 ])
    # One-Hot Encoding of categorical values
    df = pd.get_dummies(df, columns=['suburb', 'type'], drop_first=True)
    # Splitting features from label
    from sklearn.model_selection import train_test_split

    X = df.drop('price', axis=1)
    with open('ai_model/train_columns.pkl', 'wb') as file:
        pickle.dump(X.columns, file)
    y = df['price']

    # Splitting data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training a linear regression model
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    linear_model_path = 'ai_model/linear_regression_model.pkl'
    joblib.dump(model, linear_model_path)
    joblib.dump(scaler, 'ai_model/scaler.bin', compress=True)
    return 'Linear regression model updated successfully'

@app.route('/update_randomforrest_model', methods=['GET'])
def update_randomforrest_model():
    # Import joblib
    import joblib

    # Get the count of all records
    total_count = db.session.query(MelbourneHousingData).count()

    # Calculate 80% of the total count
    limit = int(total_count * 0.8)

    # Get 80% of the data
    query_result = db.session.query(MelbourneHousingData).limit(limit).all()
    # # Execute a query to get data from the table
    # query_result = db.session.query(MelbourneHousingData).all()

    # Convert the query result to a dataframe
    df = pd.DataFrame([(row.suburb, 
                        row.rooms,
                        row.type, 
                        row.price,
                        row.distance,
                        row.bathroom,
                        row.car,
                        row.landarea
                        ) for row in query_result], 
                        columns=['suburb',
                                 'rooms',
                                 'type', 
                                 'price',
                                 'distance',
                                 'bathroom',
                                 'car',
                                 'landarea',
                                 ])
    # One-Hot Encoding of categorical values
    df_encoded = pd.get_dummies(df, columns=['suburb', 'type'], drop_first=True)

    # drop nan
    df_encoded = df_encoded.dropna()

    # Splitting features from label
    from sklearn.model_selection import train_test_split

    X = df_encoded.drop('price', axis=1)
    with open('ai_model/train_columns.pkl', 'wb') as file:
        pickle.dump(X.columns, file)
    y = df_encoded['price']



    # Splitting data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training a model
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor

    # Initialize the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf_model.fit(X_train_scaled, y_train)
    
    random_forest_model_path = 'ai_model/random_forest_model.pkl'
    joblib.dump(rf_model, random_forest_model_path)
    joblib.dump(scaler, 'ai_model/scaler.bin', compress=True)
    return 'Random forest model updated successfully'

@app.route('/market_demand', methods=['POST'])
def market_demand():
    # Get the request data from the client
    data = request.json
    suburb = data['suburb']
    # Execute a query to get properties from the table for the selected suburb
    query_result = db.session.query(MelbourneHousingData).filter(MelbourneHousingData.suburb == suburb).all()
    # Convert the query result to a dataframe
    df = pd.DataFrame([(row.suburb,
                        row.address,
                        row.date) for row in query_result],
                        columns=['Suburb',
                                    'Address',
                                    'Date',
                                    ])
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Extract month and year from 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Group by month number and year to count the number of sales
    sales_per_month = df.groupby(['Year', 'Month']).size().reset_index(name='Number of Sales')

    # Map month numbers to month names
    month_map = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    sales_per_month['Month_Name'] = sales_per_month['Month'].map(month_map)
    sales_per_month['Date'] = pd.to_datetime(sales_per_month['Year'].astype(str) + '-' + sales_per_month['Month'].astype(str) + '-01')
    sales_per_month = sales_per_month.set_index('Date')
    sales_per_month_data = sales_per_month['Number of Sales']

    model = ARIMA(sales_per_month_data, order=(1,0,1))
    model_fit = model.fit()

    # Forecast for the next 6 months
    forecast = np.ceil(model_fit.forecast(steps=6)).values.astype('int')

    # Plotting the data
    sales_per_month_data.plot(figsize=(10.5,6) ,label='Historical Sales')
    plt.title('Monthly Sales Data for {suburb}'.format(suburb=suburb))
    plt.ylabel('Number of Sales')
    plt.xlabel('Date')
    plt.grid(True)

    # Plot the forecasted sales data for the next 6 months
    forecast_dates = pd.date_range(sales_per_month_data.index[-1], periods=6, freq='M')  # Creating dates for the forecasted period
    plt.plot(forecast_dates, forecast, label='Forecasted Sales', color='red')
    plt.legend()
    plt.show()
    
    # Set y-axis to display only whole numbers
    ax = plt.gca()  # Get the current Axes instance
    ax.yaxis.set_major_locator(MultipleLocator(1))  # Set locator to multiples of 1

    # Convert matplotlib figure to plotly figure
    mpl_fig = plt.gcf()
    fig = mpl_to_plotly(mpl_fig)

    return jsonify({'sales_per_month': fig.to_json(), 'forecast': forecast.tolist()})


if __name__ == '__main__':
   app.run(debug=True)
