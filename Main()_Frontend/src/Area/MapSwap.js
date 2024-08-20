import React, { useEffect, useRef, useContext, useState } from "react";
import "ol/ol.css";
import Swal from "sweetalert2";
import { Map, View } from "ol";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat, toLonLat } from "ol/proj";
import VectorSource from "ol/source/Vector";
import VectorLayer from "ol/layer/Vector";
import { Feature } from "ol";
import { Point } from "ol/geom";
import { Style, Icon, Text, Fill, Stroke } from "ol/style";
import Modify from "ol/interaction/Modify";
import "./MapSwap.css";
import { DataContext } from "../Data/DataContext";

function MapSwap({ selectedCamData, selectedAddress }) {
  const { data, setData } = useContext(DataContext);
  const userId = data?.userId;
  //const address = data?.mapcameraprovideoinfo[0].address;
  const Server_IP = process.env.REACT_APP_Server_IP;

  const mapElement = useRef(null);
  const mapRef = useRef(null);
  const markerSource = useRef(new VectorSource());
  const [markerList, setMarkerList] = useState([]);
  const [address, setAddress] = useState(selectedAddress);

  // selectedAddress가 변경될 때 address를 업데이트하는 함수
  const updateAddress = () => {
    setAddress(selectedAddress);
    console.log("Address updated to:", selectedAddress);
  };

  useEffect(() => {
    updateAddress(); // selectedAddress 변경 시 address 업데이트
  }, [selectedAddress]); // selectedAddress가 변경될 때마다 실행

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
            style: function (feature) {
              return getMarkerStyle(feature.get("name"));
            },
          }),
        ],
        view: new View({
          center: fromLonLat([126.978, 37.5665]), // 초기 위치 설정
          zoom: 12, // 초기 줌 레벨 설정
        }),
      });

      const modify = new Modify({ source: markerSource.current });
      mapRef.current.addInteraction(modify);

      modify.on("modifyend", function (event) {
        const updatedMarkers = markerSource.current
          .getFeatures()
          .map((feature) => {
            const coordinates = toLonLat(
              feature.getGeometry().getCoordinates()
            );
            return {
              name: feature.get("name"),
              latitude: coordinates[1],
              longitude: coordinates[0],
            };
          });
        setMarkerList(updatedMarkers);
      });

      mapRef.current.on("singleclick", function (event) {
        if (event.originalEvent.shiftKey && event.originalEvent.button === 0) {
          const coordinate = event.coordinate;

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
              marker.setStyle(
                new Style({
                  image: new Icon({
                    anchor: [0.5, 1],
                    src: "https://openlayers.org/en/v4.6.5/examples/data/icon.png",
                  }),
                  text: new Text({
                    text: result.value || "Unnamed",
                    offsetY: -25,
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

              const lonLat = toLonLat(coordinate);
              setMarkerList((prev) => [
                ...prev,
                {
                  name: result.value || "Unnamed",
                  latitude: lonLat[1],
                  longitude: lonLat[0],
                },
              ]);
            }
          });
        }
      });

      mapRef.current.on("singleclick", function (event) {
        if (event.originalEvent.ctrlKey && event.originalEvent.button === 0) {
          mapRef.current.forEachFeatureAtPixel(event.pixel, function (feature) {
            if (feature.getGeometry().getType() === "Point") {
              Swal.fire({
                title: `${feature.get("name")}을(를) 삭제하시겠습니까?`,
                icon: "warning",
                showCancelButton: true,
                confirmButtonText: "삭제",
                cancelButtonText: "취소",
              }).then((result) => {
                if (result.isConfirmed) {
                  markerSource.current.removeFeature(feature);

                  setMarkerList((prev) =>
                    prev.filter(
                      (marker) =>
                        marker.latitude !==
                          toLonLat(feature.getGeometry().getCoordinates())[1] ||
                        marker.longitude !==
                          toLonLat(feature.getGeometry().getCoordinates())[0]
                    )
                  );
                }
              });
            }
          });
        }
      });

      mapRef.current.on("dblclick", function (event) {
        mapRef.current.forEachFeatureAtPixel(event.pixel, function (feature) {
          if (feature.getGeometry().getType() === "Point") {
            // 현재 마커의 이름을 가져옴
            let currentName = feature.get("name");

            console.log("현재 마커의 이름: " + currentName);

            // 이름이 설정되지 않은 경우 기본값을 제공
            if (currentName === undefined || currentName === null) {
              currentName = "Unnamed"; // 기본값 설정
            }

            Swal.fire({
              title: "새로운 이름을 입력하세요:",
              input: "text",
              inputValue: currentName, // 기본값을 설정하여 undefined 방지
              showCancelButton: true,
              confirmButtonText: "확인",
              cancelButtonText: "취소",
            }).then((result) => {
              if (result.isConfirmed && result.value) {
                // 이름을 수정하여 feature에 반영
                feature.set("name", result.value);
                feature.setStyle(getMarkerStyle(result.value));
                console.log("새로운 이름: " + result.value);

                // markerList도 업데이트하여 서버에 보낼 수 있도록 설정
                setMarkerList((prev) =>
                  prev.map((marker) =>
                    marker.latitude ===
                      toLonLat(feature.getGeometry().getCoordinates())[1] &&
                    marker.longitude ===
                      toLonLat(feature.getGeometry().getCoordinates())[0]
                      ? { ...marker, name: result.value }
                      : marker
                  )
                );
              }
            });
          }
        });
      });
    }

    markerSource.current.clear();

    const loadMarkers = selectedCamData || data.mapcameraprovideoinfo;

    if (loadMarkers && Array.isArray(loadMarkers)) {
      const initialMarkers = [];
      loadMarkers.forEach((cam) => {
        if (cam.address === address) {
          const { cam_longitude, cam_latitude, cam_name } = cam;
          const marker = new Feature({
            geometry: new Point(
              fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)])
            ),
            name: cam_name,
          });

          marker.setStyle(getMarkerStyle(cam_name));

          markerSource.current.addFeature(marker);
          initialMarkers.push({
            name: cam_name,
            latitude: parseFloat(cam_latitude),
            longitude: parseFloat(cam_longitude),
          });
        }
      });

      setMarkerList(initialMarkers);

      if (loadMarkers.length > 0) {
        const { cam_longitude, cam_latitude } = loadMarkers[0];
        mapRef.current
          .getView()
          .setCenter(
            fromLonLat([parseFloat(cam_longitude), parseFloat(cam_latitude)])
          );
        mapRef.current.getView().setZoom(17);
      }
    }
  }, [selectedCamData]);

  const handleDeleteData = async () => {
    Swal.fire({
      title: "삭제하시겠습니까?",
      text: "삭제 시 모든 정보가 사라집니다.",
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "삭제",
      cancelButtonText: "취소",
    }).then(async (result) => {
      if (result.isConfirmed) {
        try {
          // userId와 address를 쿼리 파라미터로 전달
          console.log("User ID:", userId);
          console.log("Address:", address);

          const response = await fetch(
            `${Server_IP}/delete_map?user_id=${userId}&address=${address}`,
            {
              method: "DELETE",
              headers: {
                "Content-Type": "application/json",
              },
            }
          );

          if (!response.ok) {
            throw new Error("Network response was not ok");
          }

          const data = await response.json();

          // 서버에서 반환된 데이터 구조에 맞게 상태를 업데이트합니다.
          setData({
            mapcameraprovideoinfo: data.updated_map_camera_provideo_info,
          });

          // 지도에서 마커를 삭제합니다.
          markerSource.current.clear();
          setMarkerList([]);

          Swal.fire({
            title: "카메라 위치가 성공적으로 삭제되었습니다.",
            icon: "success",
            confirmButtonText: "확인",
          });
        } catch (error) {
          console.error("Error deleting map camera info:", error);
          Swal.fire({
            title: "카메라 위치 삭제 중 오류가 발생했습니다.",
            icon: "error",
            confirmButtonText: "확인",
          });
        }
      }
    });
  };

  const handleSendData = async () => {
    console.log("MapSwap Send address : " + address);
    const markerData = {
      userId: userId,
      address: address,
      markers: markerList.map((marker) => {
        console.log("Marker name:", marker.name); // name 출력
        return {
          name: marker.name,
          latitude: marker.latitude,
          longitude: marker.longitude,
        };
      }),
    };

    try {
      const response = await fetch(`${Server_IP}/update_cams`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(markerData),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data1 = await response.json();
      setData({ mapcameraprovideoinfo: data1.map_camera_info });

      Swal.fire({
        title: "카메라 위치가 성공적으로 저장되었습니다.",
        icon: "success",
        confirmButtonText: "확인",
      });
    } catch (error) {
      console.error("Error fetching map camera info:", error);
      Swal.fire({
        title: "카메라 위치 저장 중 오류가 발생했습니다.",
        icon: "error",
        confirmButtonText: "확인",
      });
    }
  };

  const getMarkerStyle = (name) => {
    return new Style({
      image: new Icon({
        anchor: [0.5, 1],
        src: "https://openlayers.org/en/v4.6.5/examples/data/icon.png",
      }),
      text: new Text({
        text: name,
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
  };

  return (
    <div className="MapSwap">
      <div className="header">
        <h2 className="title">지도 정보</h2>
        <button className="delete-button" onClick={handleDeleteData}>
          지도 삭제
        </button>
      </div>
      <div className="map-and-info">
        <div className="map-container" ref={mapElement}></div>
        <div className="camera-info">
          <button className="edit-button" onClick={handleSendData}>
            지도 수정
          </button>
          <p>카메라 추가 : Shift + 마우스 좌클릭</p>
          <p>카메라 삭제 : Ctrl + 마우스 좌클릭</p>
          <p>카메라 이동 : 마커 클릭 후 드래그</p>
          <p>카메라 이름 변경 : 마커 더블 클릭</p>
        </div>
      </div>
    </div>
  );
}

export default MapSwap;
