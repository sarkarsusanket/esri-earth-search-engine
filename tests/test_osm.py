import os
import unittest
from unittest.mock import patch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Import the functions and structures from osm.py
import osm
from schema import GEOMETRY_COL, SCORE_COL


class TestOSMSearch(unittest.TestCase):

    def setUp(self):
        """Set up dummy GeoDataFrames simulating OSM layer data."""
        # Dummy points and polygons in EPSG:4326
        self.poly_a = Polygon([(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)])
        self.point_inside = Point(1, 1)
        self.point_outside = Point(5, 5)

        # Mock POI data
        self.pois_gdf = gpd.GeoDataFrame(
            {
                "amenity": [
                    "car_rental",
                    "bicycle_rental",
                    "car_wash",
                    "parking;bicycle_parking",
                    "residential",
                ],
                "name": [
                    "Enterprise Rent-A-Car",
                    "City Bike Station",
                    "Sparkle Car Wash",
                    "Central Parking",
                    "Sunset Apartments",
                ],
                GEOMETRY_COL: [
                    self.point_inside,
                    self.point_inside,
                    self.point_outside,
                    self.point_inside,
                    self.point_outside,
                ],
            },
            crs="EPSG:4326",
        )

        # Mock Roads data
        self.roads_gdf = gpd.GeoDataFrame(
            {
                "highway": ["residential", "motorway_link", "primary"],
                GEOMETRY_COL: [
                    self.point_inside,
                    self.point_outside,
                    self.point_inside,
                ],
            },
            crs="EPSG:4326",
        )

        self.mock_osm_data = {
            "pois": self.pois_gdf,
            "roads": self.roads_gdf,
        }

        # Spatial bounding region for testing spatial filtering
        self.region_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COL: [self.poly_a]},
            crs="EPSG:4326",
        )

    # -------------------------------------------------------------------------
    # Helper & Stemming Tests
    # -------------------------------------------------------------------------

    def test_stemming(self):
        """Test plural stripping and stemming helper."""
        self.assertEqual(osm._stem("rivers"), "river")
        self.assertEqual(osm._stem("beaches"), "beach")
        self.assertEqual(osm._stem("cities"), "city")
        self.assertEqual(osm._stem("parking"), "parking")

    def test_resolve_mode_and_query(self):
        """Test argument resolution between mode and query strings."""
        # Both None
        self.assertEqual(osm._resolve_mode_and_query(None, None), ("roads", None))
        # Arg1 is mode, Arg2 is query
        self.assertEqual(
            osm._resolve_mode_and_query("pois", "car_rental"), ("pois", "car_rental")
        )
        # Arg1 is query, Arg2 is mode
        self.assertEqual(
            osm._resolve_mode_and_query("car_rental", "pois"), ("pois", "car_rental")
        )
        # Unsupported mode falls back to query with default mode
        self.assertEqual(
            osm._resolve_mode_and_query("bakery", None), ("roads", "bakery")
        )

    # -------------------------------------------------------------------------
    # Category Hit Logic Tests
    # -------------------------------------------------------------------------

    def test_category_hits_exact_compound_match(self):
        """Test that compound queries like 'car_rental' do NOT match 'bicycle_rental'."""
        hits = osm._category_hits("car_rental", self.pois_gdf, "amenity")
        self.assertIn("car_rental", hits)
        self.assertNotIn("bicycle_rental", hits)
        self.assertNotIn("car_wash", hits)

    def test_category_hits_multi_value_tag(self):
        """Test matching against semicolon-delimited multi-value tags."""
        hits = osm._category_hits("bicycle_parking", self.pois_gdf, "amenity")
        self.assertIn("parking;bicycle_parking", hits)

    def test_category_hits_stemming(self):
        """Test that plural query 'residentials' matches category 'residential'."""
        hits = osm._category_hits("residentials", self.pois_gdf, "amenity")
        self.assertIn("residential", hits)

    # -------------------------------------------------------------------------
    # Name Search Tests
    # -------------------------------------------------------------------------

    def test_name_hits(self):
        """Test string matching on the name column."""
        indices = osm._name_hits("Enterprise", self.pois_gdf)
        self.assertEqual(len(indices), 1)
        self.assertEqual(self.pois_gdf.loc[indices[0]]["name"], "Enterprise Rent-A-Car")

    # -------------------------------------------------------------------------
    # Search OSM Integration Tests
    # -------------------------------------------------------------------------

    def test_search_osm_with_spatial_filter(self):
        """Test full search pipeline with region filtering."""
        result = osm.search_osm(
            mode="pois",
            query="car_rental",
            region=self.region_gdf,
            osm_data=self.mock_osm_data,
        )

        # Expected: 1 match (car_rental point inside region_gdf)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 1)
        self.assertIn(SCORE_COL, result.columns)
        self.assertEqual(result.iloc[0]["amenity"], "car_rental")

    def test_search_osm_filtered_out_by_region(self):
        """Test query matching features that get filtered out spatially."""
        result = osm.search_osm(
            mode="pois",
            query="car_wash",  # car_wash is at point (5, 5), outside polygon (0, 0, 2, 2)
            region=self.region_gdf,
            osm_data=self.mock_osm_data,
        )

        self.assertTrue(result.empty)

    def test_search_osm_no_query_returns_all_in_region(self):
        """Test search with no query term returns all region-intersecting features."""
        result = osm.search_osm(
            mode="roads",
            query=None,
            region=self.region_gdf,
            osm_data=self.mock_osm_data,
        )

        # 2 road features fall inside region_gdf (points at 1,1)
        self.assertEqual(len(result), 2)

    def test_search_osm_unsupported_mode(self):
        """Test behavior when an invalid mode is provided."""
        result = osm.search_osm(
            mode="invalid_mode",
            query="test",
            region=None,
            osm_data=self.mock_osm_data,
        )
        self.assertTrue(result.empty)

    # -------------------------------------------------------------------------
    # Parquet Data Loader Mock Test
    # -------------------------------------------------------------------------

    @patch("os.path.isdir", return_value=True)
    @patch("os.path.isfile", return_value=True)
    @patch("geopandas.read_parquet")
    def test_load_osm_data(self, mock_read_parquet, mock_isfile, mock_isdir):
        """Test loading parquet files using mocked filesystem and GeoPandas calls."""
        mock_read_parquet.return_value = self.pois_gdf

        data = osm.load_osm_data(osm_dir="/dummy/dir", year="2026")

        self.assertIn("pois", data)
        self.assertIn("roads", data)
        self.assertEqual(mock_read_parquet.call_count, len(osm.MODE_REGISTRY))


if __name__ == "__main__":
    unittest.main()