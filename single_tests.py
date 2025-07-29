import numpy as np
import scipy.stats as stats

# Example data
data1 = np.random.normal(loc=0, scale=1, size=1000)  # Normally distributed data
data2 = np.random.normal(loc=0.5, scale=1.5, size=1000)  # Normally distributed data with different mean and variance

# Function to perform the Shapiro-Wilk test for normality
def test_normality(data):
    stat, p_value = stats.shapiro(data)
    if p_value < 0.05:
        return False, p_value  # Not normal
    else:
        return True, p_value  # Normal

# Function to decide and perform the appropriate statistical test
def perform_statistical_test(data1, data2):
    normal1, p_value1 = test_normality(data1)
    normal2, p_value2 = test_normality(data2)

    print("Data1 - Normality: {}, p-value: {:.4f}".format(normal1, p_value1))
    print("Data2 - Normality: {}, p-value: {:.4f}".format(normal2, p_value2))

    # If both datasets are normal, use Welch's T-test
    if normal1 and normal2:
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        significant = p_value < 0.05
        print("\nWelch's T-test is applied:")
        print("Significant Difference:", significant)
        print("p-value:", p_value)
    else:
        # If either dataset is not normal, use Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(data1, data2)
        significant = p_value < 0.05
        print("\nMann-Whitney U-Test is applied:")
        print("Significant Difference:", significant)
        print("p-value:", p_value)

# Running the tests
perform_statistical_test(data1, data2)
