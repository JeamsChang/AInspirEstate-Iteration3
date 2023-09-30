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
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler


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
@app.route('/browsing', methods=['GET'])
def browsing():
   print('Request for browsing page received')
   
   # Get all suburbs from database
   suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
   
   # Get max number of rooms from database
   max_rooms = db.session.query(db.func.max(MelbourneHousingData.rooms)).scalar()
   
   # Get max number of bathrooms from database
   max_bathroom = db.session.query(db.func.max(MelbourneHousingData.bathroom)).scalar()
   
   # Get max price of properties from database
   max_price = db.session.query(db.func.max(MelbourneHousingData.price)).scalar()
   
   # Get min price of properties from database
   min_price = db.session.query(db.func.min(MelbourneHousingData.price)).scalar()

   # Get max price of properties from database
   max_price = db.session.query(db.func.max(MelbourneHousingData.price)).scalar()
   
   # Get latitude and longitude of properties from the database
   coordinates = db.session.query(MelbourneHousingData.latitude, MelbourneHousingData.longitude).all()
   return render_template('browsing.html', 
                          suburbs=suburbs, 
                           max_rooms=max_rooms, 
                           max_price=max_price, 
                           max_bathroom=max_bathroom, 
                           min_price=min_price, 
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

@app.route('/get_garages', methods=['GET'])
def get_garages():
    # Get the selected suburb and type from the request parameters
    selectedSuburb = request.args.get('suburb')
    selectedType = request.args.get('type')
    selectedBedrooms = request.args.get('bedroom')
    selectedBathrooms = request.args.get('bathroom')

    # query available garages for the selected suburb, type, bedrooms and bathrooms
    garages = db.session.query(MelbourneHousingData.car).filter(MelbourneHousingData.suburb == selectedSuburb, MelbourneHousingData.type == selectedType, MelbourneHousingData.rooms == selectedBedrooms, MelbourneHousingData.bathroom == selectedBathrooms).distinct().all()
    return jsonify([garage[0] for garage in garages])


# query properties
@app.route('/browsing_post', methods=['POST'])
def browsing_post():
   # accept the request data from the client
   data = request.json
   suburb = data['suburb']
   bedrooms = data['bedrooms']
   bathrooms = data['bathrooms']
   maxPrice = data['maxPrice']

   # search the database for properties that match the search criteria
   properties = db.session.query(MelbourneHousingData).filter(MelbourneHousingData.suburb == suburb, 
                                                               MelbourneHousingData.rooms == bedrooms, 
                                                               MelbourneHousingData.bathroom == bathrooms, 
                                                               MelbourneHousingData.price <= maxPrice).all()
   
   # get the coordinates of the properties
   properties_info = [(property.latitude, property.longitude, property.rooms, property.bathroom, property.car, property.price) for property in properties]

   return jsonify(properties_info)

# suburb statistics  
@app.route('/avg', methods=['POST'])
def avg():
   data = request.json
   suburb = data['suburb']
   avg_bedrooms = db.session.query(db.func.avg(MelbourneHousingData.rooms)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_bathrooms = db.session.query(db.func.avg(MelbourneHousingData.bathroom)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_price = db.session.query(db.func.avg(MelbourneHousingData.price)).filter(MelbourneHousingData.suburb == suburb).scalar()
   avg_car = db.session.query(db.func.avg(MelbourneHousingData.car)).filter(MelbourneHousingData.suburb == suburb).scalar()
   return jsonify(
      [round(avg_bedrooms), round(avg_bathrooms), round(avg_car), round(avg_price)]
      )

# top3 properties
@app.route('/top3', methods=['POST'])
def top3():
    data = request.json
    suburb = data['suburb']
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    maxPrice = data['maxPrice']
    properties = db.session.query(MelbourneHousingData).filter(MelbourneHousingData.suburb == suburb,
                                                               MelbourneHousingData.rooms == bedrooms, 
                                                               MelbourneHousingData.bathroom == bathrooms,
                                                                MelbourneHousingData.price <= maxPrice
                                                               ).order_by(MelbourneHousingData.price).limit(3).all()
    properties_info = [(property.address, property.price) for property in properties]
    return jsonify(properties_info)

@app.route('/prediction', methods=['GET'])
def prediction():
    print('Request for prediction page received')
    # update_linear_regression_model()
    suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
    return render_template('prediction.html', suburbs=suburbs)

@app.route('/predict_price', methods=['POST'])
def predict_price():

    # Get the request data from the client
    data = request.json
    suburb = data['suburb']
    bedrooms = data['rooms']
    property_type = data['type']
    bathrooms = data['bathroom']
    car_spaces = data['car']

    # Creating a new data
    new_data = pd.DataFrame({
        'suburb': [suburb],
        'rooms': [bedrooms],
        'type': [property_type],
        'distance': [0],
        'bathroom': [bathrooms],
        'car': [car_spaces],
        'landarea': [0]
    })

    # load the model
    model = joblib.load('ai_model/linear_regression_model.pkl')

    # load the train columns
    with open('ai_model/train_columns.pkl', 'rb') as file:
        train_columns = pickle.load(file)

    # Preprocessing the new data in the same way as the training data
    scaler = joblib.load('ai_model/scaler.bin')
    new_data = pd.get_dummies(new_data, columns=['suburb', 'type'], drop_first=True).reindex(columns=train_columns, fill_value=0)
    new_data_scaled = scaler.transform(new_data)

    # Making predictions
    predicted_price = model.predict(new_data_scaled)[0]

    print(f"Expected Price: ${round(predicted_price)}")
    return jsonify(round(predicted_price))



@app.route('/intervention', methods=['GET'])
def intervention():
    print('Request for intervention page received')
    return render_template('intervention.html')

@app.route('/demand', methods=['GET'])
def demand():
    print('Request for demand page received')
    suburbs = db.session.query(MelbourneHousingData.suburb).distinct().order_by(MelbourneHousingData.suburb).all()
    return render_template('demand.html', suburbs=suburbs)

# update linear regression model in the database
@app.route('/update_linear_regression_model', methods=['POST'])
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

    # Create a bar chart with Plotly
    fig = px.line(sales_per_month, x='Month_Name', y='Number of Sales', color='Year', 
             title='Number of Sales per Month', 
             labels={'Month_Name': 'Month', 'Number of Sales': 'Number of Sales'},
             category_orders={"Month_Name": list(month_map.values())})
    
    # Set y-axis to display only whole numbers and adjust margin for x-axis
    max_sales = sales_per_month['Number of Sales'].max()
    fig.update_layout(yaxis=dict(tickvals=list(range(0, max_sales+1))))

    # Group by year to get total sales for each year
    total_sales_per_year = df.groupby('Year').size().reset_index(name='Total Sales')

    # Calculate the percentage change from the previous year
    total_sales_per_year['Percentage Change'] = round(total_sales_per_year['Total Sales'].pct_change() * 100, 2)

    # Create a new column for the year interval representation
    total_sales_per_year['Year Interval'] = (total_sales_per_year['Year'] - 1).astype(str) + '-' + total_sales_per_year['Year'].astype(str)

    # Drop the first row since we don't have percentage change data for the first year
    total_sales_per_year = total_sales_per_year.dropna()

    return jsonify({'sales_per_month': fig.to_json(), 'total_sales_per_year': total_sales_per_year.to_json(orient='records')})
    









if __name__ == '__main__':
   app.run(debug=True)
