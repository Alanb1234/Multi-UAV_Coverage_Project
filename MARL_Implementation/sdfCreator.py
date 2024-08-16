import json

def load_grid(filename):
    with open(filename, 'r') as f:
        grid = json.load(f)
        #print(grid)
    return grid

def create_sdf_optimized(grid, output_filename='world_optimized.sdf'):
    grid_height = len(grid)
    grid_width = len(grid[0])

    
    sdf_template = '''<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="world_optimized">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <plugin name='gz::sim::systems::Physics' filename='libgz-sim-physics-system.so'/>
    <plugin name='gz::sim::systems::UserCommands' filename='libgz-sim-user-commands-system.so'/>
    <plugin name='gz::sim::systems::SceneBroadcaster' filename='libgz-sim-scene-broadcaster-system.so'/>
    <plugin name='gz::sim::systems::Contact' filename='libgz-sim-contact-system.so'/>
    <plugin name='gz::sim::systems::Imu' filename='libgz-sim-imu-system.so'/>
    <plugin name='gz::sim::systems::AirPressure' filename='libgz-sim-air-pressure-system.so'/>
    <plugin name='gz::sim::systems::Sensors' filename='libgz-sim-sensors-system.so'>
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.80665</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <scene>
      <grid>false</grid>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
'''

    # Combine adjacent obstacle cells into larger blocks
    visited = [[False for _ in range(grid_width)] for _ in range(grid_height)]

    for i in range(grid_height):
        for j in range(grid_width):
            if grid[i][j] == 1 and not visited[i][j]:
                x_start = j
                y_start = i
                x_end = j
                y_end = i

                # Find the extent of this obstacle block
                while x_end + 1 < grid_width and grid[y_start][x_end + 1] == 1:
                    x_end += 1
                while y_end + 1 < grid_height and all(grid[y][x_start:x_end + 1] == [1] * (x_end - x_start + 1) for y in range(y_start, y_end + 2)):
                    y_end += 1

                # Mark these cells as visited
                for y in range(y_start, y_end + 1):
                    for x in range(x_start, x_end + 1):
                        visited[y][x] = True

                # Add this block to the SDF
                block = f'''
    <model name="block_{i}_{j}">
      <static>true</static>
      <pose>{(x_start + x_end) / 2} {(y_start + y_end) / 2} 1 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{x_end - x_start + 1} {y_end - y_start + 1} 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{x_end - x_start + 1} {y_end - y_start + 1} 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.5 0.5 0.5 1</specular>
          </material>
        </visual>
      </link>
    </model>
'''
                sdf_template += block


    # Close the SDF template
    sdf_template += '''
  </world>
</sdf>
'''

    # Write to output file
    with open(output_filename, 'w') as f:
        f.write(sdf_template)

# Usage
if __name__ == "__main__":
    grid = load_grid('LAB_world.json')
    create_sdf_optimized(grid, 'world_optimized.sdf')
