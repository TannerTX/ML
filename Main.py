from Model_Consumer import ModelProducer as Model_Producer
from reformat import Formatter

model_NM = 'Mdl-2025-07-01'
x = Model_Producer((220, 220), 'jpg', ['./Cat_Imgs', './Dog_Imgs'], './postprocessed', model_NM, 30)
# x.run()
x.predict(model_NM)
# x.clean_slate()

# formatter = Formatter((220, 220), 'jpg', './postprocessed')
# formatter.resize('./Dog_Imgs', './predict_images')
