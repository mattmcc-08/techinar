import pandas as pd
pizza_file = 'data/pizzas.csv' 
pizza_data = pd.read_csv(pizza_file)
print(pizza_data)

import math
pizza_data['area'] = [((diameter / 2) ** 2) * math.pi for diameter in pizza_data['Size']]
pizza_data['price_per_sq_inch'] = pizza_data['Price'] / pizza_data['area']
print(pizza_data)
pizza= pizza_data.sort_values(by='price_per_sq_inch', ascending=True)
print(pizza)