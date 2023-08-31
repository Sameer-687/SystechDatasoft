class Vehicle:
    def __init__(self,manufacturer,model):
        self.manufacturer=manufacturer
        self.model=model

    def get_info(self):
        return "Manufacturer of the vehicle: {} \nThe model of the vehicle: {}".format(self.manufacturer,self.model)
        #return "Manufacturer of the vehicle: {0.manufacturer} \n The model of the vehicle: {0.model}".format(self)

v1=Vehicle("Mercedes", "CLA")

print(v1.get_info())