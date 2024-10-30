# Horizon File Format
The scheduler supports two formats for horizon files:

1. AZ-ALT format (recommended):
   - First line must contain "AZ-ALT"
   - Each subsequent line contains: <azimuth> <altitude>
   - Fast loading (~0.1ms)

2. HA-DEC format (legacy support):
   - Each line contains: <hour_angle> <declination>
   - Requires coordinate conversion on load (~300ms overhead)
   - Consider converting to AZ-ALT format for better performance
