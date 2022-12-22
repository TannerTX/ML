from Model_Consumer import Model_Producer
from reformat import Formatter

model_NM = 'fullAuto'
x = Model_Producer((200, 200), 'jpg', ['./Cat_Imgs', './Dog_Imgs'], './postprocessed', model_NM, 10)
# x.run()
# x.predict(model_NM)
x.clean_slate()

# formatter = Formatter((200, 200), 'jpg', './postprocessed')
# formatter.resize('./Dog_Imgs', './predict_images')
