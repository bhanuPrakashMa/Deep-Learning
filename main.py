from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

checker = Checker(20,2)
checker.draw()
checker.show()

circle = Circle(100, 10, (50,50))
circle.draw()
circle.show()

spectrum = Spectrum(100)
spectrum.draw()
spectrum.shows()

testgen = ImageGenerator(
    r"E:\Deep Learning Lab\exercise0_material\src_to_implement\exercise_data",
    r"E:\Deep Learning Lab\exercise0_material\src_to_implement\Labels.json",
    50, [8,8,3], rotation=False, mirroring=False, shuffle=False)
testgen.next()
testgen.show()
