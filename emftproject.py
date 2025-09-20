import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests
from datetime import datetime
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

# Optional cartopy import - graceful fallback if not installed
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except Exception as e:
    print("cartopy not available or failed to import. Map animations will fallback to simple plots.")
    USE_CARTOPY = False

class IonospherePropagationSimulator:
    def __init__(self):
        # Constants
        self.EARTH_RADIUS = 6371.0  # km
        self.IONOSPHERE_HEIGHT = 350.0  # default km (approximate F-layer height)
        self.API_URL = "https://services.swpc.noaa.gov/json/f107_cm.json"
        self.CITY_DATA = self.load_city_data()

    def load_city_data(self):
        """Load city data from CSV file or create a comprehensive list."""
        city_data = {}

        # Try to load from CSV file if available
        csv_path = "world_cities.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    name = str(row.get('city', '')).lower()
                    lat = row.get('lat', None)
                    lng = row.get('lng', None)
                    if name and lat is not None and lng is not None:
                        city_data[name] = (float(lat), float(lng))
                if city_data:
                    print(f"Loaded {len(city_data)} cities from CSV")
                    return city_data
            except Exception as e:
                print(f"Error loading CSV: {e}. Using built-in city database.")

        # Fallback built-in list
        cities = {
            'new york': (40.7128, -74.0060),
            'london': (51.5074, -0.1278),
            'tokyo': (35.6895, 139.6917),
            'sydney': (-33.8688, 151.2093),
            'moscow': (55.7558, 37.6173),
            'cairo': (30.0444, 31.2357),
            'rio de janeiro': (-22.9068, -43.1729),
            'los angeles': (34.0522, -118.2437),
            'beijing': (39.9042, 116.4074),
            'mumbai': (19.0760, 72.8777),
            'paris': (48.8566, 2.3522),
            'berlin': (52.5200, 13.4050),
            'rome': (41.9028, 12.4964),
            'madrid': (40.4168, -3.7038),
            'delhi': (28.6139, 77.2090),
            'kolkata': (22.5726, 88.3639),
            'chennai': (13.0827, 80.2707),
            'bangalore': (12.9716, 77.5946),
            'singapore': (1.3521, 103.8198),
            'dubai': (25.2048, 55.2708),
            'istanbul': (41.0082, 28.9784),
            'auckland': (-36.8485, 174.7633),
            'wellington': (-41.2865, 174.7762),
            'melbourne': (-37.8136, 144.9631),
            'santiago': (-33.4489, -70.6693),
        }

        print(f"Using built-in database of {len(cities)} cities")
        return cities

    def get_city_suggestions(self, query):
        q = query.lower()
        matches = [city for city in self.CITY_DATA.keys() if q in city]
        return matches[:5]

    def get_ionospheric_data(self):
        """
        Fetch near-real-time ionospheric/solar proxy data from NOAA SWPC.
        Fall back to synthetic data if API is unavailable or format unexpected.
        """
        try:
            response = requests.get(self.API_URL, timeout=6)
            response.raise_for_status()
            data = response.json()
            # API returns list of records, find latest with f107 if present
            latest = None
            if isinstance(data, list) and len(data) > 0:
                # pick last or first that has f107
                for rec in reversed(data):
                    if rec and ('f107' in rec or 'f107a' in rec):
                        latest = rec
                        break
                if latest is None:
                    latest = data[-1]
            elif isinstance(data, dict):
                latest = data
            else:
                latest = None

            if latest:
                # Accept either 'f107' or 'f107a' depending on API structure
                solar_flux = None
                if 'f107' in latest:
                    solar_flux = float(latest['f107'])
                elif 'f107a' in latest:
                    solar_flux = float(latest['f107a'])

                if solar_flux is not None:
                    # Empirical proxy -> foF2
                    foF2 = 5.0 + (solar_flux - 70.0) * 0.05
                    foF2 = float(np.clip(foF2, 3.0, 15.0))
                    timestamp = latest.get('time_tag') or latest.get('time') or datetime.utcnow().isoformat()
                    return {'foF2': foF2, 'source': 'NOAA SWPC (proxy)', 'timestamp': timestamp, 'solar_flux': solar_flux}

            # If we reach here, fallback to synthetic
            raise ValueError("API format unexpected or missing f107")
        except Exception as e:
            print(f"API request failed or unexpected format ({e}). Using synthetic data.")
            return self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """Generate synthetic ionospheric data with realistic values."""
        hour = datetime.utcnow().hour
        # diurnal factor: higher near 10-16 utc local variation approx
        diurnal_factor = 0.7 + 0.3 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 0.5
        month = datetime.utcnow().month
        seasonal_factor = 0.8 + 0.2 * np.sin((month - 3) * np.pi / 6)
        random_factor = 0.9 + 0.2 * np.random.random()
        foF2 = 8.0 * diurnal_factor * seasonal_factor * random_factor
        return {'foF2': float(foF2), 'source': 'Synthetic', 'timestamp': datetime.utcnow().isoformat(), 'solar_flux': 70 + 30 * np.random.random()}

    def calculate_max_electron_density(self, foF2):
        """N_max in electrons per m^3 from foF2 (MHz)"""
        return (foF2 / 8.98) ** 2 * 1e10

    def calculate_critical_frequency(self, N_max):
        return 8.98 * np.sqrt(N_max / 1e10)

    def check_propagation(self, tx_frequency_mhz, foF2):
        """Return (reflection_occurs_bool, muf_mhz)"""
        muf = 3.5 * foF2
        return (tx_frequency_mhz <= muf, muf)

    def calculate_path_loss(self, distance_km, frequency_mhz):
        """FSPL + empirical ionospheric loss (dB). Guard against tiny distance."""
        d = max(distance_km, 1e-3)
        fsl = 32.4 + 20.0 * np.log10(d) + 20.0 * np.log10(frequency_mhz)
        iono_loss = 15.0 + 5.0 * np.random.random()
        return fsl + iono_loss

    def calculate_great_circle_distance(self, lat1, lon1, lat2, lon2):
        """Haversine distance in km"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return self.EARTH_RADIUS * c

    def get_coordinates_from_city(self, city_name):
        city_name_lower = city_name.strip().lower()
        if city_name_lower in self.CITY_DATA:
            return self.CITY_DATA[city_name_lower]
        suggestions = self.get_city_suggestions(city_name_lower)
        if suggestions:
            # Auto-pick first suggestion instead of interactive prompt
            chosen = suggestions[0]
            print(f"City '{city_name}' not exact. Using nearest match: {chosen.title()}")
            return self.CITY_DATA[chosen]
        print(f"City '{city_name}' not found. Using random location.")
        lat = float(np.random.uniform(-60, 60))
        lon = float(np.random.uniform(-180, 180))
        return lat, lon

    def parse_location_input(self, location_input):
        """Coordinates 'lat,lon' or city name."""
        if not isinstance(location_input, str):
            location_input = str(location_input)
        s = location_input.strip()
        try:
            if ',' in s:
                parts = [p.strip() for p in s.split(',')]
                if len(parts) == 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return lat, lon
            return self.get_coordinates_from_city(s)
        except Exception as e:
            print(f"Error parsing '{location_input}': {e}. Using random location.")
            return float(np.random.uniform(-60, 60)), float(np.random.uniform(-180, 180))

    def calculate_wave_path(self, tx_lat, tx_lon, rx_lat, rx_lon, reflection_occurs):
        """Return map_path (lon_points, lat_points), side_path (x_km, height_km), distance, reflection_point_idx"""
        distance = self.calculate_great_circle_distance(tx_lat, tx_lon, rx_lat, rx_lon)
        if distance < 1e-6:
            # Same location: trivial path
            lon_points = [tx_lon, rx_lon]
            lat_points = [tx_lat, rx_lat]
            side_x = np.array([0.0, 1.0])
            side_y = np.array([0.0, 0.0])
            return {'map_path': (lon_points, lat_points), 'side_path': (side_x, side_y), 'distance': distance, 'reflection_point_idx': None}

        # Compute intermediate great-circle points (spherical interpolation)
        c = distance / self.EARTH_RADIUS
        tx_lat_r, tx_lon_r = np.radians(tx_lat), np.radians(tx_lon)
        rx_lat_r, rx_lon_r = np.radians(rx_lat), np.radians(rx_lon)

        num_points = 200
        fractions = np.linspace(0.0, 1.0, num_points)
        lat_points = []
        lon_points = []
        sin_c = np.sin(c)
        if abs(sin_c) < 1e-9:
            for f in fractions:
                lat_points.append(np.degrees(tx_lat_r * (1 - f) + rx_lat_r * f))
                lon_points.append(np.degrees(tx_lon_r * (1 - f) + rx_lon_r * f))
        else:
            for f in fractions:
                A = np.sin((1 - f) * c) / sin_c
                B = np.sin(f * c) / sin_c
                x = A * np.cos(tx_lat_r) * np.cos(tx_lon_r) + B * np.cos(rx_lat_r) * np.cos(rx_lon_r)
                y = A * np.cos(tx_lat_r) * np.sin(tx_lon_r) + B * np.cos(rx_lat_r) * np.sin(rx_lon_r)
                z = A * np.sin(tx_lat_r) + B * np.sin(rx_lat_r)
                lat = np.arctan2(z, np.sqrt(x**2 + y**2))
                lon = np.arctan2(y, x)
                lat_points.append(np.degrees(lat))
                lon_points.append(np.degrees(lon))

        # Side view x-axis = distance along path
        side_x = np.linspace(0.0, distance, num_points)
        if reflection_occurs:
            ri = num_points // 2
            side_y = np.zeros(num_points)
            if ri > 0:
                side_y[:ri] = self.IONOSPHERE_HEIGHT * (side_x[:ri] / side_x[ri])
                side_y[ri:] = self.IONOSPHERE_HEIGHT * (1.0 - (side_x[ri:] - side_x[ri]) / (side_x[-1] - side_x[ri]))
            else:
                side_y[:] = 0.0
            reflection_idx = ri
        else:
            escape_angle_deg = 45.0
            side_y = side_x * np.tan(np.radians(escape_angle_deg))
            reflection_idx = None

        return {'map_path': (lon_points, lat_points), 'side_path': (side_x, side_y), 'distance': distance, 'reflection_point_idx': reflection_idx}

    def create_side_view_animation(self, tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs, foF2, tx_frequency_mhz, mode="day"):
        # local ionosphere height (do not override self.IONOSPHERE_HEIGHT globally)
        ionosphere_height = 300.0 if mode == "day" else 400.0

        # Recompute side path using local ionosphere height so visuals match day/night
        distance = wave_path.get('distance', 1.0)
        num_points = 200
        side_x = np.linspace(0.0, max(distance, 1.0), num_points)
        if reflection_occurs:
            ri = num_points // 2
            side_y = np.zeros(num_points)
            if ri > 0:
                side_y[:ri] = ionosphere_height * (side_x[:ri] / side_x[ri])
                side_y[ri:] = ionosphere_height * (1.0 - (side_x[ri:] - side_x[ri]) / (side_x[-1] - side_x[ri]))
            reflection_idx = ri
        else:
            escape_angle_deg = 45.0
            side_y = side_x * np.tan(np.radians(escape_angle_deg))
            reflection_idx = None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Earth surface (blue)
        earth_x = np.linspace(0.0, max(side_x), 300)
        earth_y = np.zeros_like(earth_x)
        ax.plot(earth_x, earth_y, color='blue', linewidth=2, label='Earth Surface')
        ax.fill_between(earth_x, earth_y, -1000.0, color='blue', alpha=0.08)

        # Ionosphere (red)
        iono_y = np.full_like(earth_x, ionosphere_height)
        ax.plot(earth_x, iono_y, '--', color='red', linewidth=1.5, label=f'Ionosphere (F-layer) [{mode}]')
        ax.fill_between(earth_x, iono_y, iono_y + 40.0, color='red', alpha=0.12)

        # Plot faint full propagation path for context (green, faint)
        ax.plot(side_x, side_y, color='green', linewidth=1.0, alpha=0.25, label='Propagation Path (theoretical)')

        # Animated wave marker and trail
        wave_marker, = ax.plot([], [], 'go', markersize=8, label='Wave Front')
        wave_trail, = ax.plot([], [], color='green', linewidth=2, label='Animated Propagation')

        # Info box text (Tx freq, foF2, MUF, prop status)
        reflection_text = "Reflection (communication possible)" if reflection_occurs else "Escape (no ionospheric reflection)"
        # Compute MUF for display
        muf = 3.5 * foF2
        info_text = f"Mode: {mode.capitalize()}\nTx: {tx_frequency_mhz:.2f} MHz\nfoF2: {foF2:.2f} MHz\nMUF ≈ {muf:.2f} MHz\n{reflection_text}"
        info_box = ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85), fontsize=9)

        # Legend & axis labels
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Height (km)')
        ax.set_title('Side View — Radio Wave Propagation')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # y limits to show space above ionosphere
        ax.set_ylim(-50.0, ionosphere_height + 300.0)
        ax.set_xlim(0.0, max(side_x))

        # Animation function
        def animate(i):
            n = len(side_x)
            if reflection_occurs and reflection_idx is not None:
                total_frames = 2 * reflection_idx
                frame = i % max(total_frames, 1)
                if frame < reflection_idx:
                    idx = frame
                else:
                    idx = reflection_idx - (frame - reflection_idx) - 1
                idx = int(np.clip(idx, 0, n - 1))
            else:
                idx = int(np.clip(i % n, 0, n - 1))

            # update marker and trail
            wave_marker.set_data([side_x[idx]], [side_y[idx]])
            wave_trail.set_data(side_x[:idx+1], side_y[:idx+1])

            # update info box with a small dynamic hint
            hint = "Propagation: Reflection" if reflection_occurs else "Propagation: Escape"
            info_box.set_text(f"Mode: {mode.capitalize()}\nTx: {tx_frequency_mhz:.2f} MHz\nfoF2: {foF2:.2f} MHz\nMUF ≈ {muf:.2f} MHz\n{hint}\nFrame: {idx}/{n}")
            return wave_marker, wave_trail, info_box

        frames = len(side_x) * (2 if reflection_occurs else 1)
        ani = FuncAnimation(fig, animate, frames=frames, interval=40, blit=False, repeat=True)
        return fig, ani

    def create_map_view_animation(self, tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs):
        """
        Animated map view showing great-circle path being traced.
        Uses cartopy if available, otherwise falls back to a lon-lat animated plot.
        """
        lon_points, lat_points = wave_path['map_path']

        if USE_CARTOPY:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

            # Plot TX and RX
            ax.plot(tx_lon, tx_lat, 'ro', markersize=8, transform=ccrs.Geodetic(), label='Transmitter')
            ax.plot(rx_lon, rx_lat, 'bo', markersize=8, transform=ccrs.Geodetic(), label='Receiver')

            # Animated path and moving marker
            path_line, = ax.plot([], [], color='green', linewidth=2, transform=ccrs.Geodetic(), label='Great Circle Path')
            moving_dot, = ax.plot([], [], 'go', transform=ccrs.Geodetic())

            # Safe extent around mid-area, with minimum span
            mid_lon = (tx_lon + rx_lon) / 2.0
            mid_lat = (tx_lat + rx_lat) / 2.0
            span_lon = max(abs(tx_lon - rx_lon) * 1.5, 20.0)
            span_lat = max(abs(tx_lat - rx_lat) * 1.5, 10.0)
            ax.set_extent([mid_lon - span_lon, mid_lon + span_lon, mid_lat - span_lat, mid_lat + span_lat], crs=ccrs.PlateCarree())

            ax.legend(loc='upper left')
            ax.set_title('Map View — Radio Communication Path')

            def animate(i):
                idx = int(np.clip(i, 0, len(lon_points) - 1))
                path_line.set_data(lon_points[:idx+1], lat_points[:idx+1])
                moving_dot.set_data([lon_points[idx]], [lat_points[idx]])
                return path_line, moving_dot

            ani = FuncAnimation(fig, animate, frames=len(lon_points), interval=40, blit=False, repeat=True)
            return fig, ani

        else:
            # Fallback animated lon-lat plot (no cartopy)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot([], [], '-', linewidth=2, label='Great Circle Path')
            tx_sc = ax.plot([tx_lon], [tx_lat], 'ro', label='Transmitter')[0]
            rx_sc = ax.plot([rx_lon], [rx_lat], 'bo', label='Receiver')[0]
            path_line, = ax.plot([], [], '-', color='green', linewidth=2)
            moving_dot, = ax.plot([], [], 'go')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Lon-Lat Path (Cartopy not available)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Set axis limits with safe padding
            lon_min, lon_max = min(lon_points), max(lon_points)
            lat_min, lat_max = min(lat_points), max(lat_points)
            pad_lon = max(1.0, (lon_max - lon_min) * 0.2)
            pad_lat = max(1.0, (lat_max - lat_min) * 0.2)
            ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
            ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)

            def animate(i):
                idx = int(np.clip(i, 0, len(lon_points) - 1))
                path_line.set_data(lon_points[:idx+1], lat_points[:idx+1])
                moving_dot.set_data(lon_points[idx], lat_points[idx])
                return path_line, moving_dot

            ani = FuncAnimation(fig, animate, frames=len(lon_points), interval=40, blit=False, repeat=True)
            return fig, ani

    def create_static_plots(self, tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs, foF2, tx_frequency_mhz, mode="day"):
        side_fig, side_ani = self.create_side_view_animation(tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs, foF2, tx_frequency_mhz, mode=mode)
        if isinstance(side_fig, tuple):
            side_fig = side_fig[0]
        map_result = self.create_map_view_animation(tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs)
        if isinstance(map_result, tuple):
            map_fig = map_result[0]
        else:
            map_fig = map_result
        return side_fig, map_fig

    def run_simulation(self):
        print("=" * 60)
        print("IONOSPHERIC RADIO WAVE PROPAGATION SIMULATOR")
        print("=" * 60)

        # Acquire input interactively
        tx_input = input("\nTransmitter (city or 'lat,lon'): ").strip()
        tx_lat, tx_lon = self.parse_location_input(tx_input)

        rx_input = input("\nReceiver (city or 'lat,lon'): ").strip()
        rx_lat, rx_lon = self.parse_location_input(rx_input)

        freq_input = input("\nFrequency (e.g., '7.2M', '14.2M', '23K', '0.144G' for GHz): ").strip().upper()
        tx_frequency_mhz = 10.0  # default
        try:
            if freq_input.endswith('K'):
                tx_frequency_mhz = float(freq_input[:-1]) / 1000.0
            elif freq_input.endswith('M'):
                tx_frequency_mhz = float(freq_input[:-1])
            elif freq_input.endswith('G'):
                tx_frequency_mhz = float(freq_input[:-1]) * 1000.0
            else:
                tx_frequency_mhz = float(freq_input)
        except Exception as e:
            print(f"Invalid frequency input '{freq_input}'. Using default 10.0 MHz. ({e})")
            tx_frequency_mhz = 10.0

        time_mode = input("\nDo you want to simulate 'day' or 'night' conditions? [day/night]: ").strip().lower()
        if time_mode not in ["day", "night"]:
            print("Invalid choice. Defaulting to 'day'.")
            time_mode = "day"

        iono_data = self.get_ionospheric_data()
        foF2 = float(iono_data['foF2'])
        N_max = self.calculate_max_electron_density(foF2)

        # Determine propagation
        reflection_occurs, muf = self.check_propagation(tx_frequency_mhz, foF2)
        print(f"\nMUF (approx): {muf:.2f} MHz")
        print(f"Reflection occurs: {'Yes' if reflection_occurs else 'No'} (Tx {tx_frequency_mhz:.2f} MHz)")

        # Compute wave path (map & a loose side_path, side animation recalculates with correct iono height)
        wave_path = self.calculate_wave_path(tx_lat, tx_lon, rx_lat, rx_lon, reflection_occurs)

        # Path loss (estim)
        try:
            path_loss = self.calculate_path_loss(wave_path['distance'], tx_frequency_mhz)
            print(f"Estimated path loss: {path_loss:.2f} dB")
        except Exception:
            print("Could not compute path loss (distance may be zero).")

        # Create both animations (side + map)
        try:
            side_fig, side_ani = self.create_side_view_animation(tx_lat, tx_lon, rx_lat, rx_lon,
                                                                 wave_path, reflection_occurs, foF2, tx_frequency_mhz, mode=time_mode)
            map_fig, map_ani = self.create_map_view_animation(tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs)
            # Display both figures; animation will run in each figure window / notebook cell
            plt.show()
        except Exception as e:
            print(f"Animation failed with error: {e}. Producing static plots instead.")
            fig1, fig2 = self.create_static_plots(tx_lat, tx_lon, rx_lat, rx_lon, wave_path, reflection_occurs, foF2, tx_frequency_mhz, mode=time_mode)
            if fig1 is not None:
                fig1.show()
            if fig2 is not None:
                fig2.show()

        print("\nSimulation complete.")

if __name__ == "__main__":
    simulator = IonospherePropagationSimulator()
    simulator.run_simulation()


