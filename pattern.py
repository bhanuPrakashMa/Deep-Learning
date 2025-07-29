import matplotlib.pyplot as plt
import numpy as np



class Checker:

    
    
    def __init__(self,resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution,resolution))

    
    
    
    def draw(self):

        bloc = (self.tile_size, self.tile_size)
        w_b = np.ones(bloc, dtype=int)
        b_b = w_b*0

        first_blc = np.tile(np.concatenate((b_b,w_b), axis=1), self.resolution // (2 * self.tile_size))
        second_blc = np.flip(first_blc, axis=1)
        checker = np.concatenate((first_blc, second_blc), axis=0)

        self.output = np.tile(checker, (self.resolution // (2 * self.tile_size), 1))
        return self.output.copy()
    

    
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()




class Circle:

    
    
    
    def __init__ (self,resolution, radius, pos):
        self.resolution = resolution
        self.radius = radius
        self.pos = pos
        self.output = np.zeros((resolution,resolution))

    
    
    
    def draw(self):
        x_cor, y_cor = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        ccir = np.zeros((self.resolution,self.resolution))

        ccir[(np.sqrt((x_cor - self.pos[0]) ** 2 + (y_cor - self.pos[1]) ** 2)) < self.radius] = 1

        self.output = ccir

        return self.output.copy()
    
    
    
    def show(self):
        plt.imshow(self.output)
        plt.show()



class Spectrum:

    
    def __init__(self, resolution):
        self.res =resolution
        self.output = np.zeros((resolution, resolution, 3))

    
    
    def draw(self):


        color = np.linspace(0,1, self.res)
        self.output[:,:, 0] = color[np.newaxis, :]
        self.output[:,:,1] = color[:, np.newaxis]
        self.output[:,:,2] = np.flip(self.output[:,:,0], axis=1)

        return self.output.copy()
    
    
    
    def shows(self):
        plt.imshow(self.output)
        plt.show()








