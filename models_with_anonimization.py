import initial_model as initial_model
import anonimizing_functions as anon


def test_noise():
    # sprawdzenie dla kolumny age
    train1, test1 = initial_model.read_data()
    p1 = initial_model.get_pipeline()
    train1 = anon.add_noise(train1, 'age')
    initial_model.create_model(p1, train1, test1)

    train2, test2 = initial_model.read_data()
    p2 = initial_model.get_pipeline()
    train2 = anon.add_noise(train2, 'fnlwgt', 2000)
    initial_model.create_model(p2, train2, test2)

def test_generalize():
    train1, test1 = initial_model.read_data()
    p1 = initial_model.get_pipeline()
    train1 = anon.generalize(train1, 'age')
    initial_model.create_model(p1, train1, test1)

    train2, test2 = initial_model.read_data()
    p2 = initial_model.get_pipeline()
    train2 = anon.generalize(train2, 'capital-gain')
    initial_model.create_model(p2, train2, test2)

    train3, test3 = initial_model.read_data()
    p3 = initial_model.get_pipeline()
    train3 = anon.generalize(train3, 'capital-loss')
    initial_model.create_model(p3, train3, test3)

    train4, test4 = initial_model.read_data()
    p4 = initial_model.get_pipeline()
    train4 = anon.generalize(train4, 'hours-per-week')
    initial_model.create_model(p4, train4, test4)

def test_suppress():
    train1, test1 = initial_model.read_data()
    p1 = initial_model.get_pipeline()
    train1 = anon.suppress_column(train1, 'age')
    initial_model.create_model(p1, train1, test1)

    train2, test2 = initial_model.read_data()
    p2 = initial_model.get_pipeline()
    train2 = anon.suppress_column(train2, 'race')
    initial_model.create_model(p2, train2, test2)

    train3, test3 = initial_model.read_data()
    p3 = initial_model.get_pipeline()
    train3 = anon.suppress_column(train3, 'sex')
    initial_model.create_model(p3, train3, test3)

    train4, test4 = initial_model.read_data()
    p4 = initial_model.get_pipeline()
    train4 = anon.suppress_column(train4, 'marital-status')
    initial_model.create_model(p4, train4, test4)

    train5, test5 = initial_model.read_data()
    p5 = initial_model.get_pipeline()
    train5 = anon.suppress_column(train5, 'native-country')
    initial_model.create_model(p5, train5, test5)

def test_perturb():
    train1, test1 = initial_model.read_data()
    p1 = initial_model.get_pipeline()
    train1 = anon.suppress_column(train1, 'hours-per-week')
    initial_model.create_model(p1, train1, test1)

    train2, test2 = initial_model.read_data()
    p2 = initial_model.get_pipeline()
    train2 = anon.suppress_column(train2, 'education-num')
    initial_model.create_model(p2, train2, test2)

    train3, test3 = initial_model.read_data()
    p3 = initial_model.get_pipeline()
    train3 = anon.suppress_column(train3, 'fnlwgt')
    initial_model.create_model(p3, train3, test3)

    train4, test4 = initial_model.read_data()
    p4 = initial_model.get_pipeline()
    train4 = anon.suppress_column(train4, 'age')
    initial_model.create_model(p4, train4, test4)

test_noise()
test_generalize()
test_suppress()
test_perturb()