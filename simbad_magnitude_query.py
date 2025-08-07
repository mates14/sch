from astroquery.simbad import Simbad
from astropy import units as u
from math import ceil

def get_magnitude(object_name):
    # Create a custom Simbad object with the 'flux(V)' field
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('flux(V)')

    # Query Simbad
    result_table = custom_simbad.query_object(object_name)

    if result_table is None:
        return None

    # Extract the V magnitude
    v_mag = result_table['FLUX_V'][0]

    return v_mag

def main():
    # Example usage
    object_name = "HZ Her"
    object_name = "AM Her"
    object_name = "GK Per"
    object_name = "OJ 287"
    magnitude = get_magnitude(object_name)

    if magnitude is not None:
        print(f"The V magnitude of {object_name} is: {magnitude}")
    else:
        print(f"Could not retrieve magnitude for {object_name}")

    exptime = 60*10**(( magnitude - 15.0 )/1.25) # 10 sigma
    #exptime = 60*10**(( magnitude - 13.2 )/1.25) # 30 sigma
    reqtime = exptime*14/6
    print(f"To get this object with 10 sigma you need exptime {exptime}s")
    print(f" at SBT in all 4 filters you should request {reqtime} s ({ceil(reqtime/300)}x 5 min window)")

if __name__ == "__main__":
    main()

