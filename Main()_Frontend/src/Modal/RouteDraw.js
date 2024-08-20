import React, { useEffect, useRef, useContext, useCallback } from "react";
import "ol/ol.css";
import { Map, View } from "ol";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat } from "ol/proj";
import VectorSource from "ol/source/Vector";
import VectorLayer from "ol/layer/Vector";
import { Feature } from "ol";
import { Point, LineString, Circle as CircleGeom } from "ol/geom";
import { Style, Icon, Text, Fill, Stroke } from "ol/style";
import "./RouteDraw.css";
import { MapDataContext } from "../Data/MapDataContext"; // MapDataContext를 import

function RouteDraw({ selectedCamData }) {
  const mapElement = useRef(null);
  const mapRef = useRef(null);
  const markerSource = useRef(new VectorSource());
  const routeSource = useRef(new VectorSource());
  const circleSource = useRef(new VectorSource());
  const isAnimating = useRef(false);

  // MapDataContext에서 필요한 값들을 가져옵니다.
  const {
    updateMapInfo,
    MapInfo,
    triggerRouteAnimation,
    resetRouteAnimation,
    isInitialMapInfo,
  } = useContext(MapDataContext);

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
          new VectorLayer({
            source: routeSource.current,
          }),
          new VectorLayer({
            source: circleSource.current,
          }),
        ],
        view: new View({
          center: fromLonLat([126.978, 37.5665]),
          zoom: 12,
        }),
      });
    }

    if (isAnimating.current) return; // 애니메이션 중에는 스왑을 방지

    markerSource.current.clear();
    routeSource.current.clear();
    circleSource.current.clear(); // 맵 스왑 시 모든 그린 것들 지우기

    // MapInfo를 기반으로 맵의 초기 위치 설정
    if (
      MapInfo &&
      MapInfo.last_camera_latitude &&
      MapInfo.last_camera_longitude
    ) {
      mapRef.current
        .getView()
        .setCenter(
          fromLonLat([
            MapInfo.last_camera_longitude,
            MapInfo.last_camera_latitude,
          ])
        );
      mapRef.current.getView().setZoom(15);
    }

    // MapInfo.cam에서 카메라 정보 가져오기
    if (MapInfo && Array.isArray(MapInfo.cam)) {
      MapInfo.cam.forEach((cam) => {
        const { cam_longitude, cam_latitude, cam_name, clip_time } = cam;
        const marker = new Feature({
          geometry: new Point(
            fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)])
          ),
          name: cam_name,
        });

        marker.setStyle(
          new Style({
            image: new Icon({
              anchor: [0.5, 1],
              src: "https://openlayers.org/en/v4.6.5/examples/data/icon.png",
            }),
            text: new Text({
              text: `${cam_name}\n${clip_time}`, // 이름과 시간 정보를 함께 표시
              offsetY: -35, // 마커 위에 띄우기 위해 Y 오프셋을 더 크게 설정
              fill: new Fill({
                color: "#000",
              }),
              stroke: new Stroke({
                color: "#fff",
                width: 2,
              }),
            }),
          })
        );

        markerSource.current.addFeature(marker);
      });

      if (MapInfo.cam.length > 0) {
        const { cam_longitude, cam_latitude } = MapInfo.cam[0];
        mapRef.current
          .getView()
          .setCenter(
            fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)])
          );
        mapRef.current.getView().setZoom(18);
      }
    }
  }, [MapInfo]); // MapInfo가 업데이트될 때도 useEffect를 실행

  const drawCircleAtLastCamera = useCallback(() => {
    if (
      MapInfo &&
      MapInfo.last_camera_latitude &&
      MapInfo.last_camera_longitude &&
      MapInfo.radius
    ) {
      const centerCoord = fromLonLat([
        MapInfo.last_camera_longitude,
        MapInfo.last_camera_latitude,
      ]);
      const maxRadius = MapInfo.radius;
      let currentRadius = 0;
      const duration = 10000000;
      const startTime = Date.now();

      const animateCircle = () => {
        const elapsedTime = Date.now() - startTime;
        const progress = Math.min(elapsedTime / duration, 1);
        currentRadius = progress * maxRadius;

        const circleFeature = new Feature({
          geometry: new CircleGeom(centerCoord, currentRadius),
        });

        circleSource.current.clear();
        circleSource.current.addFeature(circleFeature);

        circleFeature.setStyle(
          new Style({
            stroke: new Stroke({
              color: "#ff0000", // 원의 테두리 색상
              width: 1.5, // 원의 테두리 두께
            }),
            fill: new Fill({
              color: "rgba(255, 0, 0, 0.1)", // 원의 내부 색상 (투명도 포함)
            }),
          })
        );

        if (progress < 1 && triggerRouteAnimation) {
          requestAnimationFrame(animateCircle);
        } else {
          isAnimating.current = false; // 애니메이션 종료
        }
      };

      isAnimating.current = true; // 애니메이션 시작
      animateCircle();
    } else {
      circleSource.current.clear();
    }
  }, [MapInfo, triggerRouteAnimation]);

  const drawRoute = useCallback(() => {
    if (MapInfo && MapInfo.order_data && MapInfo.order_data.length > 0) {
      const features = markerSource.current.getFeatures();
      const coordinates = MapInfo.order_data
        .split("/")
        .map((name) => {
          const feature = features.find((f) => f.get("name") === name);
          return feature ? feature.getGeometry().getCoordinates() : null;
        })
        .filter((coord) => coord !== null);

      if (coordinates.length > 1) {
        routeSource.current.clear();
        circleSource.current.clear();

        let currentSegmentIndex = 0;
        let progress = 0;

        const drawNextSegment = () => {
          if (
            triggerRouteAnimation &&
            currentSegmentIndex < coordinates.length - 1
          ) {
            const start = coordinates[currentSegmentIndex];
            const end = coordinates[currentSegmentIndex + 1];

            const currentX = start[0] + (end[0] - start[0]) * progress;
            const currentY = start[1] + (end[1] - start[1]) * progress;

            const lineFeature = new Feature({
              geometry: new LineString([start, [currentX, currentY]]),
              style: new Style({
                stroke: new Stroke({
                  color: "#ff0000",
                  width: 3,
                }),
              }),
            });

            if (progress === 0) {
              routeSource.current.addFeature(lineFeature);
            } else {
              const currentFeatures = routeSource.current.getFeatures();
              const lastFeature = currentFeatures[currentFeatures.length - 1];
              if (lastFeature) {
                lastFeature
                  .getGeometry()
                  .setCoordinates([start, [currentX, currentY]]);
              }
            }

            progress += 0.01;

            if (progress >= 1) {
              if (currentSegmentIndex === coordinates.length - 2) {
                console.log("경로그리기 완료");

                drawCircleAtLastCamera();
                return;
              }
              progress = 0;
              currentSegmentIndex++;
            }

            if (currentSegmentIndex < coordinates.length - 1 || progress < 1) {
              requestAnimationFrame(drawNextSegment);
            } else {
              isAnimating.current = false; // 애니메이션 종료
            }
          }
        };

        isAnimating.current = true; // 애니메이션 시작
        drawNextSegment();
      }
    }
  }, [MapInfo, drawCircleAtLastCamera, triggerRouteAnimation]);

  useEffect(() => {
    if (triggerRouteAnimation) {
      drawRoute();
      resetRouteAnimation();
    }
  }, [triggerRouteAnimation, drawRoute, resetRouteAnimation]);

  return (
    <div className="RouteDraw">
      <div className="map-container" ref={mapElement}></div>
    </div>
  );
}

export default RouteDraw;
