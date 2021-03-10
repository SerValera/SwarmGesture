import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


loss = (0.3546, 0.0674, 0.0534, 0.0483, 0.0410, 0.0388, 0.0350, 0.0330, 0.0297, 0.0303, 0.0270, 0.0270, 0.0254, 0.0249)

acc = (0.8896, 0.9791, 0.9827, 0.9840, 0.9862, 0.9870, 0.9882, 0.9888, 0.9899, 0.9897, 0.9908, 0.9910, 0.9914, 0.9916)

val_loss = (0.0388, 0.0250, 0.0183, 0.0121, 0.0144, 0.0141, 0.0109, 0.0086, 0.0120, 0.0110, 0.0101, 0.0082, 0.0064, 0.0065)

val_acc = (0.9876, 0.9917, 0.9940, 0.9953, 0.9946, 0.9943, 0.9951, 0.9965, 0.9953, 0.9962, 0.9959, 0.9970, 0.9975, 0.9974)

epoch = np.array(range(1,15))

grid_x_ticks = np.arange(0, 15)


plt.figure()
plt.plot(epoch, acc, 'b', label='accuracy')
plt.plot(epoch, acc, 'b.')
#
# plt.plot(epoch, val_loss, 'r',  label='val loss')
# plt.plot(epoch, val_loss, 'r.')

plt.xlabel("Epoch of learning")
plt.ylabel("Average accuracy")
plt.ylim([0.86, 1])
plt.xticks(grid_x_ticks)


plt.legend()
plt.title('Accuracy')

plt.grid()
plt.show()


plt.figure()
plt.plot(epoch, loss, 'r', label='loss')
plt.plot(epoch, loss, 'r.')

plt.title('Loss')
plt.xticks(grid_x_ticks)
plt.ylabel("Loss")
plt.xlabel("Epoch of learning")
plt.legend()
plt.grid()
plt.show()