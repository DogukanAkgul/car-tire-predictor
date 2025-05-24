import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('/content/car_tire_design_updated_20000_dataset.csv')

# Encode categorical variables
label_encoders = {}
for column in ['Brand', 'Car_Type', 'Tire_Design']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target variable
X = df.drop('Tire_Design', axis=1)
y = df['Tire_Design']

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['Tire_Design'].classes_)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Classification Report:\n{report}")

# Function for prediction based on user input
def make_prediction():
    # Collect user input
    brand = input(f"Choose a Brand {list(label_encoders['Brand'].classes_)}: ")
    car_type = input(f"Choose a Car Type {list(label_encoders['Car_Type'].classes_)}: ")
    weight = int(input("Enter the Weight (kg) (e.g., 1500): "))
    wheel_size = int(input("Enter the Wheel Size (in) (e.g., 16): "))
    engine_size = float(input("Enter the Engine Size (L) (e.g., 2.5): "))
    car_age = int(input("Enter the Car Age (years) (e.g., 5): "))

    # Prepare the input data
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Car_Type': [car_type],
        'Weight_kg': [weight],
        'Wheel_Size_in': [wheel_size],
        'Engine_Size_L': [engine_size],
        'Car_Age_Years': [car_age]
    })

    # Encode and scale input data
    for column in ['Brand', 'Car_Type']:
        input_data[column] = label_encoders[column].transform(input_data[column])
    input_data_scaled = scaler.transform(input_data)

    # Predict tire design
    prediction = model.predict(input_data_scaled)
    tire_design = label_encoders['Tire_Design'].inverse_transform(prediction)
    print(f"Predicted Tire Design: {tire_design[0]}")

# Call the prediction function
make_prediction()
