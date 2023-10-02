# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2020-2021 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Module for downloading and preparing data from the ESGF servers 
to be used in atlite.
"""
import xarray as xr
from pyesgf.search import SearchConnection
import logging
import dask
from ..gis import maybe_swap_spatial_dims
import numpy as np
import pandas as pd
from wrf import interplevel
import datetime

# Null context for running a with statements without any context
try:
    from contextlib import nullcontext
except ImportError:
    # for Python verions < 3.7:
    import contextlib

    @contextlib.contextmanager
    def nullcontext():
        yield


logger = logging.getLogger(__name__)
features = {
    # "wind": ["wnd", "z", "ua", "va", "ta", "hus", "wnd_azimuth"],
    "wind": ["wnd", "z", "ua", "va", "ta", "hus", "pa"],
    "influx": ["influx", "outflux"],
    # "temperature": ["temperature", "humidity", "pressure"],
    "temperature": ["temperature","humidity", "pressure"],
    # "runoff": ["runoff"], # removing because focus on wind
}
crs = 4326
dask.config.set({"array.slicing.split_large_chunks": True})


# def search_ESGF(esgf_params, url="https://esgf-data.dkrz.de/esg-search"):
def search_ESGF(esgf_params, url="https://esgf-node.ipsl.upmc.fr/esg-search"):
    # def search_ESGF(esgf_params, url="https://esgf.ceda.ac.uk/esg-search"): # updated - cmip
    # def search_ESGF(esgf_params, url="https://esgf-node.llnl.gov/esg-search"): # updated - cmip
    conn = SearchConnection(url, distrib=True)
    ctx = conn.new_context(latest=True, **esgf_params)
    if ctx.hit_count == 0:
        ctx = ctx.constrain(frequency=esgf_params["frequency"] + "Pt")
        if ctx.hit_count == 0:
            raise (ValueError("No results found in the ESGF_database"))
    latest = ctx.search()[0]
    return latest.file_context().search()


def get_data_influx(esgf_params_all, cutout, **retrieval_params):
    """Get influx data for given retrieval parameters."""
    esgf_params = {
        "source_id": esgf_params_all["source_id"],
        "variant_label": esgf_params_all["variant_label"],
        "experiment_id": esgf_params_all["experiment_id"],
        "project": esgf_params_all["project"],
        "frequency": esgf_params_all["frequency_solar"],
    }

    attr = esgf_params

    coords = cutout.coords
    bounds = cutout.bounds
    times = coords["time"].to_index()  # to slice the time coords

    # starting and ending dates and hours
    time_start = datetime.datetime(
        times[0].year, times[0].month, times[0].day, times[0].hour
    )
    time_end = datetime.datetime(
        times[-1].year, times[-1].month, times[-1].day, times[-1].hour
    )

    drsds = retrieve_data(esgf_params, coords, variables=["rsds"], **retrieval_params)
    rsds = _rename_and_fix_coords(cutout, drsds)
    drsus = retrieve_data(esgf_params, coords, variables=["rsus"], **retrieval_params)
    rsus = _rename_and_fix_coords(cutout, drsus)
    rsds = rsds.sel(
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )
    rsus = rsus.sel(
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )

    ds = rsds
    ds["rsus"] = rsus.rsus

    for variable in list(ds.keys()):
        if variable not in ["rsds", "rsus"]:
            ds = ds.drop_vars(variable)

    ds = ds.rename({"rsds": "influx", "rsus": "outflux"})
    attr["variables"] = ["influx", "outflux"]
    attr.pop("variable")
    ds.attrs = attr
    return ds


def sanitize_inflow(ds):
    """Sanitize retrieved inflow data."""
    ds["influx"] = ds["influx"].clip(min=0.0)
    return ds


def get_data_wind(esgf_params_all, cutout, **retrieval_params):
    """Get wind for given retrieval parameters"""
    esgf_params = {
        "source_id": esgf_params_all["source_id"],
        "variant_label": esgf_params_all["variant_label"],
        "experiment_id": esgf_params_all["experiment_id"],
        "project": esgf_params_all["project"],
        "frequency": esgf_params_all["frequency_wind"],
        "table_id": esgf_params_all["table_id_wind"],
    }

    attr = esgf_params

    coords = cutout.coords
    bounds = cutout.bounds
    times = coords["time"].to_index()  # to slice the time coords
    # retrieve ua and va speed components for different pressure levels
    du = retrieve_data(esgf_params, coords, variables=["ua"], **retrieval_params)
    u = _rename_and_fix_coords(cutout, du)
    dv = retrieve_data(esgf_params, coords, variables=["va"], **retrieval_params)
    v = _rename_and_fix_coords(cutout, dv)

    if "ps" in u.data_vars:
        levs = slice(1.0, 0.9)
    else:
        levs = slice(50, 200)

    # starting and ending dates and hours
    time_start = datetime.datetime(
        times[0].year, times[0].month, times[0].day, times[0].hour
    )
    time_end = datetime.datetime(
        times[-1].year, times[-1].month, times[-1].day, times[-1].hour
    )

    # create a slice of data based on altitude and cutout
    u = u.sel(
        lev=levs,
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )
    v = v.sel(
        lev=levs,
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )

    # Calculate wind speed
    wspd = np.sqrt(u.ua * u.ua + v.va * v.va)

    # Condition to know if sigma-level or model-level
    if "ps" in u.data_vars:
        # parameters
        Rd = 287.058
        g = 9.80665
        eps = 0.62198
        # obtaining ta and hus
        dt = retrieve_data(esgf_params, coords, variables=["ta"], **retrieval_params)
        t = _rename_and_fix_coords(cutout, dt)
        t = t.sel(
            lev=levs,
            time=slice(time_start, time_end),
            x=slice(bounds[0], bounds[2]),
            y=slice(bounds[1], bounds[3]),
        )

        dq = retrieve_data(esgf_params, coords, variables=["hus"], **retrieval_params)
        q = _rename_and_fix_coords(cutout, dq)
        q = q.sel(
            lev=levs,
            time=slice(time_start, time_end),
            x=slice(bounds[0], bounds[2]),
            y=slice(bounds[1], bounds[3]),
        )

        tv = t.ta * ((q.hus + eps) / (eps * (1.0 + q.hus)))
        # ap = u_mid.ap
        b = u.b
        ps = t.ps
        if "p0" in u.data_vars:
            print("Hybrid Sigma-level")
            ap = u.a * u.p0
        else:
            print("Sigma-level")
            ap = u.ap
        p = ap + b * ps  # pressure at full model levels

        p = p.transpose("time", "lev", "y", "x")
        p0 = p.isel(lev=0)
        z0 = -(Rd / g) * tv.isel(lev=0) * np.log(p0 / ps)
        z = z0

        klev = len(p.lev)
        for k in range(1, klev):
            z1 = z0 - (Rd / g) * 0.5 * (tv.isel(lev=k) + tv.isel(lev=k - 1)) * np.log(
                p.isel(lev=k) / p.isel(lev=k - 1)
            )
            z = xr.concat([z, z1], "lev")
            z0 = z1

        z = z.assign_coords(lev=p.lev)
        z = z.transpose("time", "lev", "y", "x")
    else:
        print("Model-level")
        _, a = xr.broadcast(
            wspd, wspd.lev
        )  ##z = a + (b-1.) * orog  # height relative to ground
        z = a + (u.b - 1) * u.orog

    ds = u
    ds["wnd"] = wspd
    ds["z"] = z
    ds["va"] = v.va
    ds["ta"] = t.ta
    ds["hus"] = q.hus
    ds["pa"] = p

    # azimuth = np.arctan2(ds["ua"], ds["va"])
    # ds["wnd_azimuth"] = azimuth.where(azimuth >= 0, azimuth + 2 * np.pi)

    for variable in list(ds.keys()):
        # if variable not in ["ua", "va", "wnd", "z", "ta", "hus", "wnd_azimuth"]:
        if variable not in ["ua", "va", "wnd", "z", "ta", "hus", "pa"]:
            
            ds = ds.drop_vars(variable)

    # attr["variables"] = ["wind speed lvl", "z", "ua", "va", "ta", "hus", "wnd_azimuth"]
    attr["variables"] = ["wind speed lvl", "z", "ua", "va", "ta", "hus", "pa"]
    attr.pop("variable")
    ds.attrs = attr

    return ds


def get_data_temperature(esgf_params_all, cutout, **retrieval_params):
    """Get temperature and humidity for given retrieval parameters."""

    esgf_params = {
        "source_id": esgf_params_all["source_id"],
        "variant_label": esgf_params_all["variant_label"],
        "experiment_id": esgf_params_all["experiment_id"],
        "project": esgf_params_all["project"],
        "frequency": esgf_params_all["frequency_temp"],
    }

    attr = esgf_params

    coords = cutout.coords
    bounds = cutout.bounds
    times = coords["time"].to_index()  # to slice the time coords

    # starting and ending dates and hours
    time_start = datetime.datetime(
        times[0].year, times[0].month, times[0].day, times[0].hour
    )
    time_end = datetime.datetime(
        times[-1].year, times[-1].month, times[-1].day, times[-1].hour
    )

    # obtaining temperature at surface level
    dtas = retrieve_data(esgf_params, coords, variables=["tas"], **retrieval_params)
    tas = _rename_and_fix_coords(cutout, dtas)
    tas = tas.sel(
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )

    # obtaining humidity at surface level
    dqs = retrieve_data(esgf_params, coords, variables=["huss"], **retrieval_params)
    qs = _rename_and_fix_coords(cutout, dqs)
    qs = qs.sel(
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )

    # obtaining pressure at surface level
    dps = retrieve_data(esgf_params, coords, variables=["ps"], **retrieval_params)
    ps = _rename_and_fix_coords(cutout, dps)
    ps = ps.sel(
        time=slice(time_start, time_end),
        x=slice(bounds[0], bounds[2]),
        y=slice(bounds[1], bounds[3]),
    )

    ds = tas
    ds = ds.rename({"tas":"temperature"})
    ds["temperature"] = tas.tas
    ds["pressure"] = ps.ps
    ds["humidity"] = qs.huss

    for variable in list(ds.keys()):
        if variable not in ["temperature", "humidity", "pressure"]:
            ds = ds.drop_vars(variable)
    attr["variables"] = ["temperature", "humidity", "pressure"]
    # attr["variables"] = ["temperature","humidity"]
    attr.pop("variable")
    ds.attrs = attr

    return ds


def _year_in_file(time_range, years):
    """
    Find which file contains the requested years
    Parameters:
        time_range: str
            fmt YYYYMMDD-YYYYMMDD
        years: list
    """
    time_range = time_range.split(".")[0]
    s_year = int(time_range.split("-")[0][:4])
    e_year = int(time_range.split("-")[1][:4])
    date_range = pd.date_range(str(s_year), str(e_year), freq="AS")
    if s_year == e_year and e_year in years:
        return True
    elif date_range.year.isin(years).any() == True:
        return True
    else:
        return False


def retrieve_data(esgf_params, coords, variables, chunks=None, tmpdir=None, lock=None):
    """
    Download data from egsf database
    """
    time = coords["time"].to_index()
    years = time.year.unique()
    dsets = []
    if lock is None:
        lock = nullcontext()
    with lock:
        for variable in variables:
            esgf_params["variable"] = variable
            search_results = search_ESGF(esgf_params)
            files = [
                f.opendap_url
                for f in search_results
                if _year_in_file(f.opendap_url.split("_")[-1], years)
            ]
            # dsets.append(xr.open_mfdataset(files, chunks=chunks, concat_dim=["time"]))
            dsets.append(
                xr.open_mfdataset(
                    files, chunks=chunks or {}, combine="nested", concat_dim=["time"]
                )
            )  # added - cmip
    ds = xr.merge(dsets)
    # ds.attrs = {**ds.attrs, **esgf_params}
    ds.attrs = {**esgf_params}  # Modifying attributes - cmip
    return ds


def _rename_and_fix_coords(cutout, ds, add_lon_lat=True, add_ctime=False):
    """Rename 'longitude' and 'latitude' columns to 'x' and 'y' and fix roundings.
    Optionally (add_lon_lat, default:True) preserves latitude and longitude
    columns as 'lat' and 'lon'.
    CMIP specifics; shift the longitude from 0..360 to -180..180. In addition
    CMIP sometimes specify the time in the center of the output intervall this shifted to the beginning.
    """
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
    ds.lon.attrs["valid_max"] = 180
    ds.lon.attrs["valid_min"] = -180
    ds = ds.sortby("lon")
    ds = ds.rename({"lon": "x", "lat": "y"})
    #dt = cutout.dt
    ds = maybe_swap_spatial_dims(ds)
    if add_lon_lat:
        ds = ds.assign_coords(lon=ds.coords["x"], lat=ds.coords["y"])
    if add_ctime:
        ds = ds.assign_coords(ctime=ds.coords["time"])

    # shift averaged data to beginning of bin
    if isinstance(ds.time[0].values, np.datetime64) == False:
        if xr.CFTimeIndex(ds.time.values).calendar == "360_day":
            from xclim.core.calendar import convert_calendar

            ds = convert_calendar(ds, cutout.data.time, align_on="year")
        else:
            ds = ds.assign_coords(
                time=xr.CFTimeIndex(ds.time.values).to_datetimeindex(unsafe=True)
            )
    return ds


def get_data(cutout, feature, tmpdir, lock=None, **creation_parameters):
    """
    Retrieve data from the ESGF CMIP database.
    This front-end function downloads data for a specific feature and formats
    it to match the given Cutout.
    Parameters
    ----------
    cutout : atlite.Cutout
    feature : str
        Name of the feature data to retrieve. Must be in
        `atlite.datasets.cmip.features`
    tmpdir : str/Path
        Directory where the temporary netcdf files are stored.
    **creation_parameters :
        Additional keyword arguments. The only effective argument is 'sanitize'
        (default True) which sets sanitization of the data on or off.
    Returns
    -------
    xarray.Dataset
        Dataset of dask arrays of the retrieved variables.
    """
    coords = cutout.coords
    sanitize = creation_parameters.get("sanitize", True)
    if cutout.esgf_params == None:
        raise (ValueError("ESGF search parameters not provided"))
    else:
        esgf_params = cutout.esgf_params
    if esgf_params.get("frequency") == None:
        if cutout.dt == "H":
            freq = "h"
        elif cutout.dt == "3H":
            freq = "3hr"
        elif cutout.dt == "6H":
            freq = "6hr"
        elif cutout.dt == "D":
            freq = "day"
        elif cutout.dt == "M":
            freq = "mon"
        elif cutout.dt == "Y":
            freq = "year"
        else:
            raise (ValueError(f"{cutout.dt} not valid time frequency in CMIP"))
    else:
        freq = esgf_params.get("frequency")
    esgf_params["frequency"] = freq
    chunks = {"time": 10}
    retrieval_params = {"chunks": chunks, "tmpdir": tmpdir, "lock": lock}
    func = globals().get(f"get_data_{feature}")
    logger.info(f"Requesting data for feature {feature}...")
    ds = func(esgf_params, cutout, **retrieval_params)
    # cutout.data = cutout.data.assign_coords(x=ds.coords["x"], y=ds.coords["y"]) # added
    # ds = ds.sel(time=coords["time"])
    # bounds = cutout.bounds
    # ds = ds.sel(x=slice(bounds[0], bounds[2]), y=slice(bounds[1], bounds[3]))
    # ds = ds.interp({"x": cutout.data.x, "y": cutout.data.y})
    if globals().get(f"sanitize_{feature}") != None and sanitize:
        sanitize_func = globals().get(f"sanitize_{feature}")
        ds = sanitize_func(ds)

    return ds
