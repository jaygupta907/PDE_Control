import matplotlib.pyplot as plt
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_lineplot

# SETUP SIMULATION
input_reader = InputReader("couette.json", "numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)

# RUN SIMULATION
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

# LOAD DATA
path = sim_manager.output_writer.save_path_domain
quantities = ["density", "velocityX","velocityY", "pressure"]
cell_centers, cell_sizes, times, data_dict = load_data(path, quantities)



plt.figure()
plt.imshow(data_dict['velocityX'][-1,:,:,0]**2+data_dict['velocityY'][-1,:,:,0]**2)
plt.show()

# PLOT
# nrows_ncols = (1,3)
# create_lineplot(data_dict, cell_centers, times, nrows_ncols=nrows_ncols, axis="y", values=[0.0, 0.0], interval=100)

# fig, ax = plt.subplots(ncols=3)
# ax[0].plot(cell_centers[1], data_dict["density"][-1  ,0,:,0])
# ax[1].plot(cell_centers[1], data_dict["velocityX"][-1,0,:,0])
# ax[1].plot(cell_centers[1], 0.1*cell_centers[1], color="black", linestyle="--")
# ax[2].plot(cell_centers[1], data_dict["pressure"][-1 ,0,:,0])
# plt.show()