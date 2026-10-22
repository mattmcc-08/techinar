import pandas as pd
pizza_file = 'data/pizzas.csv' 
pizza_data = pd.read_csv(pizza_file)
print(pizza_data)