from numpy import *


def computer_error_for_line_given_points(b, m, points):
    # initialize error at 0
    total_error = 0
    # for every point
    for i in range(0, len(points)):
        # get the x value
        x = points[i, 0]
        # get the y value
        y = points[i, 1]
        # get the difference, square it, add it to the total
        total_error += (y - (m * x + b)) ** 2

    # Get the average
    return total_error / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    # Gradient descent
    for i in range(num_iterations):
        # update b and m with more accurate b and m using gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    # return optimal b and m
    return [b,m]


# finds the lowest point in slope because that's where error is the smallest (line of best fit)
def step_gradient(b_current, m_current, points, learning_rate):
    # starting points for our gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        # direction with respect to b and m
        # computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += (2/N) * x * (y - ((m_current * x) + b_current))

    # update our b and m values using our partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def run():
    # Step 1, Collect data
    # genfromtext gets data file and separates by comma (create data file)
    points = genfromtxt('data.csv', delimiter=',')

    # Step 2 - define our hyperparameters
    # learning_rate - how fast should our model converge
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Step 3 - train our model
    print('starting gradient descent at b = {0}, m={1}, error={2}'.format(initial_b, initial_m,
                                                                          computer_error_for_line_given_points(
                                                                              initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('ending gradient descent at b = {1}, m={2}, error={3}'.format(num_iterations, b, m,
                                                                        computer_error_for_line_given_points(b, m,
                                                                                                             points)))


if __name__ == '__main__':
    run()
