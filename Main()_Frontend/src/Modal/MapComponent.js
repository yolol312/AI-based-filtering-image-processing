import React, { useEffect, useRef } from "react";
import "ol/ol.css";
import Swal from "sweetalert2";
import { Map, View } from "ol";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat, toLonLat } from "ol/proj";
import { Feature } from "ol";
import { Point, LineString } from "ol/geom";
import VectorSource from "ol/source/Vector";
import VectorLayer from "ol/layer/Vector";
import { Style, Icon, Text, Fill, Stroke } from "ol/style";
import Modify from "ol/interaction/Modify";

const MapComponent = ({ center, zoom, route, onMarkerUpdate }) => {
  const mapElement = useRef(null);
  const mapRef = useRef(null);
  const markerSource = useRef(new VectorSource());
  const routeSource = useRef(new VectorSource());

  useEffect(() => {
    const iconStyle = (feature) =>
      new Style({
        image: new Icon({
          anchor: [0.5, 1],
          src: "https://openlayers.org/en/v4.6.5/examples/data/icon.png",
        }),
        text: new Text({
          text: feature.get("name") || "",
          offsetY: -25,
          fill: new Fill({
            color: "#000",
          }),
          stroke: new Stroke({
            color: "#fff",
            width: 2,
          }),
        }),
      });

    const markerLayer = new VectorLayer({
      source: markerSource.current,
      style: iconStyle,
    });

    const routeLayer = new VectorLayer({
      source: routeSource.current,
      style: new Style({
        stroke: new Stroke({
          color: "#ff0000",
          width: 3,
        }),
        image: new Icon({
          src: "data:image/svg+xml;base64,<svg>...</svg>", // 화살표 아이콘 추가
          anchor: [0.5, 0.5],
          rotateWithView: true,
        }),
      }),
    });

    mapRef.current = new Map({
      target: mapElement.current,
      layers: [
        new TileLayer({
          source: new OSM(),
        }),
        markerLayer,
        routeLayer,
      ],
      view: new View({
        center: fromLonLat(center),
        zoom: zoom,
      }),
    });

    // 마커 추가
    mapRef.current.on("click", function (event) {
      if (event.originalEvent.shiftKey && event.originalEvent.button === 0) {
        const coordinate = event.coordinate;

        // SweetAlert2를 사용하여 마커 이름을 입력받음
        Swal.fire({
          title: "핀의 이름을 입력하세요:",
          input: "text",
          inputPlaceholder: "마커 이름 입력",
          showCancelButton: true,
          confirmButtonText: "확인",
          cancelButtonText: "취소",
        }).then((result) => {
          if (result.isConfirmed && result.value) {
            const marker = new Feature({
              geometry: new Point(coordinate),
              name: result.value || "Unnamed",
            });
            markerSource.current.addFeature(marker);

            // 마커 업데이트 콜백 호출
            const updatedMarkers = markerSource.current
              .getFeatures()
              .map((f) => ({
                name: f.get("name"),
                latitude: toLonLat(f.getGeometry().getCoordinates())[1],
                longitude: toLonLat(f.getGeometry().getCoordinates())[0],
              }));
            onMarkerUpdate(updatedMarkers);
          }
        });
      }
    });

    // 마커 삭제
    mapRef.current.on("click", function (event) {
      if (event.originalEvent.ctrlKey && event.originalEvent.button === 0) {
        mapRef.current.forEachFeatureAtPixel(event.pixel, function (feature) {
          if (feature.getGeometry().getType() === "Point") {
            Swal.fire({
              title: "이 핀을 삭제하시겠습니까?",
              icon: "warning",
              showCancelButton: true,
              confirmButtonText: "삭제",
              cancelButtonText: "취소",
            }).then((result) => {
              if (result.isConfirmed) {
                markerSource.current.removeFeature(feature);

                // 마커 업데이트 콜백 호출
                const updatedMarkers = markerSource.current
                  .getFeatures()
                  .map((f) => ({
                    name: f.get("name"),
                    latitude: toLonLat(f.getGeometry().getCoordinates())[1],
                    longitude: toLonLat(f.getGeometry().getCoordinates())[0],
                  }));
                onMarkerUpdate(updatedMarkers);
              }
            });
          }
        });
      }
    });

    // 마커 이동 및 수정 인터랙션 추가
    const modify = new Modify({ source: markerSource.current });
    mapRef.current.addInteraction(modify);

    // 마커 이동 시 위치 업데이트
    modify.on("modifyend", function (event) {
      const updatedMarkers = markerSource.current
        .getFeatures()
        .map((feature) => {
          const coordinates = toLonLat(feature.getGeometry().getCoordinates());
          return {
            name: feature.get("name"),
            latitude: coordinates[1],
            longitude: coordinates[0],
          };
        });
      onMarkerUpdate(updatedMarkers);
    });

    // 마커 이름 변경
    mapRef.current.on("dblclick", function (event) {
      mapRef.current.forEachFeatureAtPixel(event.pixel, function (feature) {
        if (feature.getGeometry().getType() === "Point") {
          Swal.fire({
            title: "새로운 이름을 입력하세요:",
            input: "text",
            inputValue: feature.get("name"),
            showCancelButton: true,
            confirmButtonText: "확인",
            cancelButtonText: "취소",
          }).then((result) => {
            if (result.isConfirmed && result.value) {
              feature.set("name", result.value);

              // 마커 업데이트 콜백 호출
              const updatedMarkers = markerSource.current
                .getFeatures()
                .map((f) => ({
                  name: f.get("name"),
                  latitude: toLonLat(f.getGeometry().getCoordinates())[1],
                  longitude: toLonLat(f.getGeometry().getCoordinates())[0],
                }));
              onMarkerUpdate(updatedMarkers);
            }
          });
        }
      });
    });

    return () => mapRef.current.setTarget(null);
  }, [center, zoom, onMarkerUpdate]);

  useEffect(() => {
    if (mapRef.current) {
      const view = mapRef.current.getView();
      view.setCenter(fromLonLat(center));
      view.setZoom(zoom);
    }
  }, [center, zoom]);

  useEffect(() => {
    if (route.length > 0) {
      const features = markerSource.current.getFeatures() || [];
      const coordinates = route
        .map((name) => {
          const feature = features.find((f) => f.get("name") === name);
          return feature ? feature.getGeometry().getCoordinates() : null;
        })
        .filter((coord) => coord !== null);

      if (coordinates.length > 1) {
        const lineFeature = new Feature({
          geometry: new LineString(coordinates),
          style: new Style({
            image: new Icon({
              src: "data:image/svg+xml;base64,<svg>...</svg>",
              rotateWithView: true,
            }),
          }),
        });
        routeSource.current.clear();
        routeSource.current.addFeature(lineFeature);
      }
    }
  }, [route]);

  return (
    <div
      id="map"
      ref={mapElement}
      style={{ width: "100%", height: "100%" }}
    ></div>
  );
};

export default MapComponent;
