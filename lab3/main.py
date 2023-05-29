from utils import zipfile
import model

zipfile.unarchive('lab03.zip')

model.generate('num_guess.model', 'mnist_train.csv', 28)

model.test('num_guess.model', 'mnist_test.csv')

model.predict('num_guess.model', 'digits')