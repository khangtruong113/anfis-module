from models.GenericModels import GenericModels

if __name__ == '__main__':
    model = GenericModels()
    a: int = model.linear(1, 2)
    print(a)
