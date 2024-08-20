import React, { useContext, useState, useEffect } from "react";
import Box from "@mui/material/Box";
import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import { DataContext } from "../Data/DataContext";
import { PersonDataContext } from "../Data/PersonDataContext";

import "./SidebarLeft.css";

const SidebarLeft = ({
  onAddressSelect,
  onVideoSelect,
  onPersonClick,
  onAddressChange,
}) => {
  const { data } = useContext(DataContext);
  const { setPersonData } = useContext(PersonDataContext);
  const [selectedAddress, setSelectedAddress] = useState(null);

  const Server_IP = process.env.REACT_APP_Server_IP;
  const userId = data?.userId;

  useEffect(() => {}, [selectedAddress]); // selectedAddress가 변경될 때마다 useEffect 실행

  const handleAddressClick = (address) => {
    setSelectedAddress(address); // 상태 업데이트

    const selectedCameras = data.mapcameraprovideoinfo.filter(
      (cam) => cam.address === address // 상태 업데이트 후 address를 바로 사용
    );

    onAddressSelect(selectedCameras, address); // selectedAddress 대신 address 사용
    setPersonData([]);

    // 선택된 주소 변경 시 onAddressChange 호출
    if (onAddressChange) {
      onAddressChange(address);
    }
  };

  const handleTreeItemClick = async (provideoname, filterID) => {
    handleVideoNameClick(provideoname, filterID);
  };
  const handleVideoNameClick = async (filterID) => {
    const payload = {
      user_id: userId,
      filter_id: filterID,
    };
    const queryParams = new URLSearchParams(payload).toString();
    const videoUrl = `${Server_IP}/stream_video?${queryParams}`;

    console.log("Selected Video URL:", videoUrl);

    try {
      const response = await fetch(videoUrl, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Server response:", data);

        // onVideoSelect에 비디오 URL 전달
        onVideoSelect([], data.video_urls);
      } else {
        console.error("Server responded with an error:", response.status);
      }
    } catch (error) {
      console.error("Error sending data to server:", error);
    }
  };

  const handlepersonclick = async (filterId, personID) => {
    try {
      const response = await fetch(
        `${Server_IP}/select_person?filter_id=${filterId}&person_id=${personID}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.ok) {
        const result = await response.json();

        // PersonDataContext에 받아온 데이터 저장
        const dataWithFilterId = result.person_info.map((person) => ({
          ...person,
          filter_id: filterId,
        }));
        setPersonData(dataWithFilterId);

        // Person 정보를 추가로 onPersonClick에 전달 (필요한 경우)
        if (onPersonClick && typeof onPersonClick === "function") {
          onPersonClick(filterId); // onPersonClick을 호출
        }
      } else {
        const error = await response.json();
        console.error("Error fetching person data:", error.error);
      }
    } catch (error) {
      console.error("Exception while fetching person data:", error);
    }
  };

  const groupData = (data) => {
    const grouped = {};
    (data || []).forEach((video) => {
      if (!grouped[video.address]) {
        grouped[video.address] = {};
      }

      // 필터 정보 중 null이 아닌 값들만 사용하여 filterKey를 생성
      const filterParts = [];
      if (video.filter_age) filterParts.push(video.filter_age);
      if (video.filter_clothes) filterParts.push(video.filter_clothes);
      if (video.filter_color) filterParts.push(video.filter_color);
      if (video.filter_gender) filterParts.push(video.filter_gender);

      const filterKey =
        filterParts.length > 0
          ? filterParts.join(", ")
          : "No Filter Information";

      const label = video.bundle_name ? video.bundle_name : filterKey;

      if (!grouped[video.address][label]) {
        grouped[video.address][label] = {};
      }

      // person_id를 키로 하여 중복 제거
      if (!grouped[video.address][label][video.person_id]) {
        grouped[video.address][label][video.person_id] = video;
      }
    });

    // 중복 제거된 person들을 다시 배열로 변환
    Object.keys(grouped).forEach((address) => {
      Object.keys(grouped[address]).forEach((label) => {
        grouped[address][label] = Object.values(grouped[address][label]);
      });
    });

    return grouped;
  };

  const groupedVideos = groupData(data.mapcameraprovideoinfo);

  return (
    <Box className="SidebarLeft">
      <SimpleTreeView
        aria-label="file system navigator"
        defaultCollapseIcon={<ExpandMoreIcon />}
        defaultExpandIcon={<ChevronRightIcon />}
        className="tree-view"
      >
        {Object.keys(groupedVideos).map((address, addressIndex) => (
          <TreeItem
            key={addressIndex}
            itemId={`address-${addressIndex}`}
            label={address}
            onClick={() => handleAddressClick(address)}
            className={
              selectedAddress === address ? "tree-item-selected" : "tree-item"
            }
          >
            {Object.keys(groupedVideos[address])
              .filter((label) => label !== "No Filter Information")
              .map((label, labelIndex) => (
                <TreeItem
                  key={`label-${addressIndex}-${labelIndex}`}
                  itemId={`label-${addressIndex}-${labelIndex}`}
                  label={label}
                  onClick={() => {
                    const video = groupedVideos[address][label][0];
                    handleVideoNameClick(video.filter_id);
                  }}
                >
                  {groupedVideos[address][label].map((video, videoIndex) => (
                    <TreeItem
                      key={`person-${addressIndex}-${labelIndex}-${videoIndex}`}
                      itemId={`person-${addressIndex}-${labelIndex}-${videoIndex}`}
                      label={`Person${video.person_id || "Unknown"}`}
                      onClick={() =>
                        handlepersonclick(video.filter_id, video.person_id)
                      }
                    />
                  ))}
                </TreeItem>
              ))}
          </TreeItem>
        ))}
      </SimpleTreeView>
    </Box>
  );
};

export default SidebarLeft;
