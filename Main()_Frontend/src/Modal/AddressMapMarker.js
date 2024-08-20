import React, { useState, useContext } from "react";
import MapComponent from "./MapComponent";
import Swal from "sweetalert2";
import { DataContext } from "../Data/DataContext";
import "./AddressMapMarker.css";

function AddressMapMarker({ onBack, onClose }) {
  const [mapCenter, setMapCenter] = useState([126.978, 37.5665]); // 초기 서울 좌표
  const [zoom, setZoom] = useState(12); // 초기 줌 레벨
  const [route] = useState([]); // 경로를 저장할 상태
  const [markerList, setMarkerList] = useState([]); // 캠 리스트 상태 추가
  const [searchedLocation, setSearchedLocation] = useState(null); // 검색된 위치 상태 추가

  const { data, setData } = useContext(DataContext); // DataContext에서 data 가져오기
  const userId = data.userId; // data에서 userId 추출
  const Server_IP = process.env.REACT_APP_Server_IP;
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

  // 주소 전송시 맵 화면을 업데이트를 하는 함수
  const handleAddressSearch = async () => {
    const userInput = document.getElementById("address-input").value;

    if (!userInput) {
      Swal.fire({
        icon: "warning",
        title: "경고",
        text: "주소를 입력해 주세요.",
      });
      return;
    }

    const address = userInput;

    try {
      // 주소를 위도와 경도로 변환
      const geoResponse = await fetch(
        `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(
          address
        )}&key=${apiKey}`
      );
      const geoData = await geoResponse.json();

      if (geoData.results && geoData.results.length > 0) {
        const location = geoData.results[0].geometry.location;
        const latitude = location.lat;
        const longitude = location.lng;

        // 지도 업데이트
        setMapCenter([longitude, latitude]);
        setZoom(16); // 새로운 위치로 이동할 때 줌 레벨을 16으로 설정

        // 검색된 위치 저장
        setSearchedLocation({ address, latitude, longitude });
      } else {
        await Swal.fire({
          icon: "warning",
          title: "경고",
          text: "해당 주소를 찾을 수 없습니다.",
        });
      }
    } catch (error) {
      console.error("주소 검색 실패:", error);
      await Swal.fire({
        icon: "error",
        title: "실패",
        text: "주소 검색에 실패했습니다.",
      });
    }
  };

  //주소 정보와 캠이름 정보를 전송 이벤트
  const handleSendData = async () => {
    if (!searchedLocation) {
      Swal.fire({
        icon: "warning",
        title: "경고",
        text: "주소를 먼저 검색해 주세요.",
      });
      return;
    }

    if (markerList.length === 0) {
      Swal.fire({
        icon: "warning",
        title: "경고",
        text: "마커를 추가해 주세요.",
      });
      return;
    }

    try {
      const { address, latitude, longitude } = searchedLocation;

      const addressData = {
        address: address,
        map_latitude: latitude,
        map_longitude: longitude,
        user_id: userId,
      };

      const markerData = markerList.map((marker) => ({
        name: marker.name,
        latitude: marker.latitude,
        longitude: marker.longitude,
      }));

      console.log("전송 데이터:", JSON.stringify(addressData)); // 콘솔에 전송 데이터 로그 출력

      // 주소 데이터 전송
      const addressResponse = await fetch(`${Server_IP}/upload_maps`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(addressData),
      });

      if (!addressResponse.ok) {
        throw new Error("주소 전송 실패");
      }

      const addressResult = await addressResponse.json();
      console.log("주소 전송 응답:", addressResult);

      console.log("전송할 JSON 데이터:", JSON.stringify(markerData, null, 2));

      // 마커 데이터 전송
      const markerResponse = await fetch(`${Server_IP}/upload_cams`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(markerData),
      });

      if (!markerResponse.ok) {
        throw new Error("마커 리스트 전송 실패");
      }

      const markerResult = await markerResponse.json();
      console.log("마커 리스트 전송 응답:", markerResult);

      await Swal.fire({
        icon: "success",
        title: "성공",
        text: "모든 데이터가 성공적으로 전송되었습니다.",
      });

      fetchMapCameraInfo();
      onClose(); // 모달 닫기
    } catch (error) {
      console.error("데이터 전송 실패:", error);
      await Swal.fire({
        icon: "error",
        title: "실패",
        text: "데이터 전송에 실패했습니다.",
      });
    }
  };

  // 받아온 데이터를 반환을 해서 DataContext 에 저장을 해주는 이벤트
  const fetchMapCameraInfo = async () => {
    if (!userId) {
      console.error("User ID is not available");
      return;
    }

    try {
      // fetch를 사용하여 서버에서 데이터를 가져옵니다.
      const response = await fetch(
        `${Server_IP}/update_pro_video?user_id=${userId}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data1 = await response.json();
      console.log(
        "받아온 map_camera_provideo_info:",
        data1.map_camera_provideo_info
      ); // 콘솔에 데이터 출력
      // 가져온 데이터를 DataContext에 업데이트합니다.
      setData({ mapcameraprovideoinfo: data1.map_camera_provideo_info });
    } catch (error) {
      console.error("Error fetching map camera info:", error);
    }
  };

  const handleMarkerUpdate = (updatedMarkers) => {
    setMarkerList(updatedMarkers);

    // 지도 중심 업데이트 (마지막 마커로 설정)
    if (updatedMarkers.length > 0) {
      const lastMarker = updatedMarkers[updatedMarkers.length - 1];
      setMapCenter([lastMarker.longitude, lastMarker.latitude]);
      setZoom(16); // 마커 추가 시 줌 레벨을 16으로 설정
    }
  };

  return (
    <div>
      <div className="label-container">
        <label>
          카메라 추가 : Shift + 마우스 좌클릭 　|　카메라 삭제 : Ctrl + 마우스
          좌클릭 　|　카메라 이동 : 마커 클릭 후 드래그　|　카메라 이름 변경 :
          마커 더블 클릭
        </label>
      </div>
      <div className="address-map-marker">
        <div className="map-AddressMapMarker-container">
          <MapComponent
            center={mapCenter}
            zoom={zoom}
            route={route}
            onMarkerUpdate={handleMarkerUpdate} // 마커 업데이트 시 호출
          />
        </div>

        <div className="address-input-container">
          <h2>주소 입력</h2>
          <input
            id="address-input"
            type="text"
            placeholder="여기에 주소를 입력하세요."
          />
          <button
            className="address-submit-button"
            onClick={handleAddressSearch}
          >
            주소 검색
          </button>
          <div className="cam-list-container">
            <h3>카메라 목록</h3>
            <table>
              <thead>
                <tr>
                  <th>카메라 이름</th>
                  <th>위도</th>
                  <th>경도</th>
                </tr>
              </thead>
              <tbody>
                {markerList.map((marker, index) => (
                  <tr key={index}>
                    <td>{marker.name}</td> {/* 마커 이름 */}
                    <td>{marker.latitude.toFixed(4)}</td> {/* 위도 */}
                    <td>{marker.longitude.toFixed(4)}</td> {/* 경도 */}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="Map-modal-button-container">
          <button className="Map-modal-button-close" onClick={handleSendData}>
            완료
          </button>
        </div>
      </div>
    </div>
  );
}

export default AddressMapMarker;
