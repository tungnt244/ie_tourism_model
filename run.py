import anago
from anago.reader import load_data_and_labels
x_train, y_train = load_data_and_labels('./data/train.txt')
x_valid, y_valid = load_data_and_labels('./data/dev.txt')
x_test, y_test = load_data_and_labels('./data/test.txt')
model = anago.Sequence()#.load('./models')
model.train(x_train, y_train, x_valid, y_valid)
model.save(dir_path='./models')
model.eval(x_test, y_test)