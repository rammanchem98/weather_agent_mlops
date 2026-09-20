terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = "project-f0ff9de1-dfa0-4dca-957"
  region  = "europe-west2"
}

resource "google_compute_router" "nat_router" {
  name    = "qdrant-nat-router"
  network = "default"
  region  = "europe-west2"
}

resource "google_compute_router_nat" "nat_config" {
  name                               = "qdrant-nat-config"
  router                             = google_compute_router.nat_router.name
  region                             = "europe-west2"
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "allow-iap-ssh"
  network = "default"
  direction = "INGRESS"
  allow {
    protocol = "tcp"
    #ports    = ["22" , "6333"]
    ports    = ["22"]
  }
  source_ranges = ["35.235.240.0/20"]
}

resource "google_compute_firewall" "allow_qdrant_internal" {
  name    = "allow-qdrant-internal"
  network = "default"
  direction = "INGRESS"
  allow {
    protocol = "tcp"
    ports    = ["6333", "6334"]
  }
  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_instance" "qdrant_vm" {
  name         = "qdrant-vm"
  zone         = "europe-west2-b"
  machine_type = "e2-small"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
    }
  }

  attached_disk {
    source = google_compute_disk.qdrant_data.self_link
  }

  network_interface {
    network = "default"
  }

  metadata_startup_script = replace(<<-EOT
    #!/bin/bash
    curl -fsSL https://get.docker.com | sh
    mkdir -p /mnt/qdrant_data
    docker run -d -p 6333:6333 -v /mnt/qdrant_data:/qdrant/storage qdrant/qdrant
    EOT
  , "\r\n", "\n")
}

resource "google_compute_disk" "qdrant_data" {
  name = "qdrant-data-disk"
  zone = "europe-west2-b"
  size = 20
  type = "pd-standard"
}

resource "google_vpc_access_connector" "qdrant_connector" {
  name          = "qdrant-connector"
  region        = "europe-west2"
  network       = "default"
  ip_cidr_range = "10.8.0.0/28"
}

output "qdrant_vm_internal_ip" {
  value = google_compute_instance.qdrant_vm.network_interface[0].network_ip
}