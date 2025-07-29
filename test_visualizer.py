import pandas as pd
from visualizer import plot_regression

def test_plot_regression():
    # Valid data
    df = pd.DataFrame({
        'rank': [1, 2, 3, 4, 5],
        'worldwide_gross': [1000000, 2000000, 1500000, 3000000, 2500000]
    })
    result = plot_regression(df, 'rank', 'worldwide_gross', title='Test Plot')
    print('Valid data result:', result[:30])

    # Missing column
    df2 = pd.DataFrame({'rank': [1, 2, 3]})
    result2 = plot_regression(df2, 'rank', 'worldwide_gross')
    print('Missing column result:', result2)

    # All NaN after conversion
    df3 = pd.DataFrame({'rank': ['a', 'b', 'c'], 'worldwide_gross': ['x', 'y', 'z']})
    result3 = plot_regression(df3, 'rank', 'worldwide_gross')
    print('All NaN result:', result3)

if __name__ == '__main__':
    test_plot_regression()
