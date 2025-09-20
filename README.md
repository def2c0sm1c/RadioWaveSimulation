It's a Python-simulation of how the ionosphere affects radio wave propagation day/night. Visualizes reflection, absorption, and whether the transmitted signal reaches the receiver or escapes to space based on user-input frequency and transmitter/receiver locations.Ionospheric Radio Wave Propagation Simulator.

Features

---Synthetic Ionospheric Modeling

Due to the unavailability of a working real-time API, the simulation uses fully synthetic data for ionospheric conditions.

Code for fetching real-time data from NOAA SWPC is included but falls back to synthetic data when unavailable.


---Radio Wave Propagation Analysis

Calculates Maximum Usable Frequency (MUF) to determine if reflection occurs.

Estimates path loss (dB) for the given transmitter-receiver pair.


---Supported Locations

Built-in database of 25 major cities worldwide with latitude and longitude.

Users can input a city name or coordinates in the form lat,lon.

Limitation: The program only works for these predefined cities; input outside this set defaults to a random location or may fail gracefully.


---Visualizations

Side view animation: Shows the height profile of the radio wave along its path.

Map view animation: Animated great-circle path on a world map using Cartopy.

Fallback static plots are generated if animation fails or Cartopy is unavailable.


---Interactive Inputs

Transmitter and receiver city names or coordinates.

Transmission frequency (kHz, MHz, or GHz).

Day/night simulation mode to visualize ionospheric height differences.


🔹 Technologies & Libraries

Python 3.12

NumPy, Matplotlib, Pandas, Requests

Cartopy (optional, for map animation)

Standard Python libraries: Datetime, OS, Warnings


🔹 How to Use

1.Clone the repository:

git clone https://github.com/yourusername/ionosphere-propagation-simulator.git
cd ionosphere-propagation-simulator


2.Install dependencies:

pip install numpy matplotlib pandas requests cartopy


3.Run the simulation:

python emftproject.py


4.Follow prompts:

Enter transmitter and receiver (city or lat,lon).

Enter frequency (e.g., 14.2M for 14.2 MHz).

Choose day or night simulation.


5.View results:

Animated side view of wave propagation.

Animated map view of global path (if Cartopy is installed).

Static plots are generated as a fallback.


🔹 Example

Transmitter: Chennai

Receiver: London

Frequency: 14.2M

Mode: day


-- Limitations

Only works for 25 built-in cities. Other cities may produce random coordinates or fail gracefully.

Fully synthetic ionospheric data is used due to lack of real-time API access.

Cartopy is optional; without it, map animations use simple lon-lat plots.


--- Applications

Educational tool for understanding HF/VHF radio propagation.

Amateur radio enthusiasts can visualize potential long-distance communication.

Demonstrates ionospheric effects for space weather or atmospheric studies.
