import React, { useEffect, useRef, useContext } from 'react';
import 'ol/ol.css';
import { Map, View } from 'ol';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import { fromLonLat } from 'ol/proj';
import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import { Feature } from 'ol';
import { Point } from 'ol/geom';
import { Style, Icon, Text, Fill, Stroke } from 'ol/style';
import './MapSwap.css';
import { DataContext } from '../Data/DataContext';

function MapSwap({ selectedCamData }) {
  const { data } = useContext(DataContext); // DataContext에서 데이터 가져오기
  const mapElement = useRef(null);
  const mapRef = useRef(null);
  const markerSource = useRef(new VectorSource());

  useEffect(() => {
    if (!mapRef.current) {
      mapRef.current = new Map({
        target: mapElement.current,
        layers: [
          new TileLayer({
            source: new OSM(),
          }),
          new VectorLayer({
            source: markerSource.current,
          }),
        ],
        view: new View({
          center: fromLonLat([126.978, 37.5665]),
          zoom: 12,
        }),
      });
    }

    markerSource.current.clear();

    if (selectedCamData && Array.isArray(selectedCamData)) {
      selectedCamData.forEach((selectedCam) => {
        // DataContext에서 가져온 data와 selectedCamData 비교
        const matchedCam = data.mapCameraInfo.find(
          (cam) =>
            cam.cam_name === selectedCam.cam_name &&
            cam.address === selectedCam.address &&
            cam.cam_latitude === selectedCam.cam_latitude &&
            cam.cam_longitude === selectedCam.cam_longitude,
        );

        if (matchedCam) {
          const { cam_longitude, cam_latitude, cam_name } = matchedCam;
          const marker = new Feature({
            geometry: new Point(fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)])),
            name: cam_name,
          });

          marker.setStyle(
            new Style({
              image: new Icon({
                anchor: [0.5, 1],
                src: 'https://openlayers.org/en/v4.6.5/examples/data/icon.png',
              }),
              text: new Text({
                text: cam_name,
                offsetY: -25,
                fill: new Fill({
                  color: '#000',
                }),
                stroke: new Stroke({
                  color: '#fff',
                  width: 2,
                }),
              }),
            }),
          );

          markerSource.current.addFeature(marker);
        }
      });

      if (selectedCamData.length > 0) {
        // 첫 번째 카메라 위치로 지도 센터를 설정
        const { cam_longitude, cam_latitude } = selectedCamData[0];
        mapRef.current
          .getView()
          .setCenter(fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)]));
        mapRef.current.getView().setZoom(15);
      }
    }
  }, [selectedCamData, data]);

  return (
    <div className="MapSwap">
      <h2 className="title">지도 정보</h2>
      <div
        className="map-container"
        ref={mapElement}
        style={{ width: '100%', height: '400px' }}
      ></div>
    </div>
  );
}

export default MapSwap;
