#include "rpi.h"
#include "fat32.h"
#include "fat32-helpers.h"
#include "pi-sd.h"

// Print extra tracing info when this is enabled.  You can and should add your
// own.
static int trace_p = 1; 
static int init_p = 0;

static unsigned CLUSTER_MASK = 0x0FFFFFFF;

fat32_boot_sec_t boot_sector;


fat32_fs_t fat32_mk(mbr_partition_ent_t *partition) {
  demand(!init_p, "the fat32 module is already in use\n");
  // TODO: Read the boot sector (of the partition) off the SD card.
  output("%d, %d\n", partition->lba_start, partition->nsec);
  boot_sector = *(fat32_boot_sec_t *)pi_sec_read(partition->lba_start, 1);

  // TODO: Verify the boot sector (also called the volume id, `fat32_volume_id_check`)
  fat32_volume_id_check(&boot_sector);

  // TODO: Read the FS info sector (the sector immediately following the boot
  // sector) and check it (`fat32_fsinfo_check`, `fat32_fsinfo_print`)
  assert(boot_sector.info_sec_num == 1);
  struct fsinfo *fsinfo_sector = (struct fsinfo *)pi_sec_read(partition->lba_start + 1, 1);

  fat32_fsinfo_check(fsinfo_sector);

  fat32_volume_id_print("boot sector", &boot_sector);
  fat32_fsinfo_print("fsinfo sector", fsinfo_sector);

  // END OF PART 2
  // The rest of this is for Part 3:

  // TODO: calculate the fat32_fs_t metadata, which we'll need to return.
  unsigned lba_start = partition->lba_start; // from the partition
  unsigned fat_begin_lba = lba_start + boot_sector.reserved_area_nsec; // the start LBA + the number of reserved sectors
  unsigned cluster_begin_lba = fat_begin_lba + boot_sector.nfats * boot_sector.nsec_per_fat; // the beginning of the FAT, plus the combined length of all the FATs
  unsigned sec_per_cluster = boot_sector.sec_per_cluster; // from the boot sector
  unsigned root_first_cluster = boot_sector.first_cluster; // from the boot sector
  unsigned n_entries = boot_sector.nsec_per_fat * boot_sector.bytes_per_sec / (sizeof(uint32_t)); // from the boot sector
  
  /*
   * TODO: Read in the entire fat (one copy: worth reading in the second and
   * comparing).
   *
   * The disk is divided into clusters. The number of sectors per
   * cluster is given in the boot sector byte 13. <sec_per_cluster>
   *
   * The File Allocation Table has one entry per cluster. This entry
   * uses 12, 16 or 28 bits for FAT12, FAT16 and FAT32.
   *
   * Store the FAT in a heap-allocated array.
   */
  uint32_t *fat = pi_sec_read(fat_begin_lba, boot_sector.nsec_per_fat);

  // Create the FAT32 FS struct with all the metadata
  fat32_fs_t fs = (fat32_fs_t) {
    .lba_start = lba_start,
      .fat_begin_lba = fat_begin_lba,
      .cluster_begin_lba = cluster_begin_lba,
      .sectors_per_cluster = sec_per_cluster,
      .root_dir_first_cluster = root_first_cluster,
      .fat = fat,
      .n_entries = n_entries,
  };

  if (trace_p) {
    trace("begin lba = %d\n", fs.fat_begin_lba);
    trace("cluster begin lba = %d\n", fs.cluster_begin_lba);
    trace("sectors per cluster = %d\n", fs.sectors_per_cluster);
    trace("root dir first cluster = %d\n", fs.root_dir_first_cluster);
  }

  init_p = 1;
  return fs;
}

// Given cluster_number, get lba.  Helper function.
static uint32_t cluster_to_lba(fat32_fs_t *f, uint32_t cluster_num) {
  assert(cluster_num >= 2);
  // TODO: calculate LBA from cluster number, cluster_begin_lba, and
  // sectors_per_cluster
  unsigned lba = f->cluster_begin_lba + (cluster_num - 2) * f->sectors_per_cluster;
  if (trace_p) trace("cluster %d to lba: %d\n", cluster_num, lba);
  return lba;
}

pi_dirent_t fat32_get_root(fat32_fs_t *fs) {
  demand(init_p, "fat32 not initialized!");
  // TODO: return the information corresponding to the root directory (just
  // cluster_id, in this case)
  
  return (pi_dirent_t) {
    .name = "",
      .raw_name = "",
      .cluster_id = fs->root_dir_first_cluster, 
      .is_dir_p = 1,
      .nbytes = 0,
  };
}

// Given the starting cluster index, get the length of the chain.  Helper
// function.
uint32_t get_cluster_chain_length(fat32_fs_t *fs, uint32_t start_cluster) {
  // TODO: Walk the cluster chain in the FAT until you see a cluster where
  // `fat32_fat_entry_type(cluster) == LAST_CLUSTER`.  Count the number of
  // clusters.

  uint32_t cluster = start_cluster;
  uint32_t length = 0;
  while (fat32_fat_entry_type(cluster) != LAST_CLUSTER) {
      cluster = fs->fat[cluster];
      length++;
  }

  return length;
}

// Given the starting cluster index, read a cluster chain into a contiguous
// buffer.  Assume the provided buffer is large enough for the whole chain.
// Helper function.
static void read_cluster_chain(fat32_fs_t *fs, uint32_t start_cluster, uint8_t *data) {
  // TODO: Walk the cluster chain in the FAT until you see a cluster where
  // fat32_fat_entry_type(cluster) == LAST_CLUSTER.  For each cluster, copy it
  // to the buffer (`data`).  Be sure to offset your data pointer by the
  // appropriate amount each time.
  uint32_t cluster = start_cluster;
  while (fat32_fat_entry_type(cluster) != LAST_CLUSTER) {
      pi_sd_read(data, cluster_to_lba(fs, cluster), fs->sectors_per_cluster);
      cluster = fs->fat[cluster];
      data += fs->sectors_per_cluster * boot_sector.bytes_per_sec;
  }
}

// Converts a fat32 internal dirent into a generic one suitable for use outside
// this driver.
static pi_dirent_t dirent_convert(fat32_dirent_t *d) {
  pi_dirent_t e = {
    .cluster_id = fat32_cluster_id(d),
    .is_dir_p = d->attr == FAT32_DIR,
    .nbytes = d->file_nbytes,
  };
  // can compare this name
  memcpy(e.raw_name, d->filename, sizeof d->filename);
  // for printing.
  fat32_dirent_name(d,e.name);
  return e;
}

// Gets all the dirents of a directory which starts at cluster `cluster_start`.
// Return a heap-allocated array of dirents.
static fat32_dirent_t *get_dirents(fat32_fs_t *fs, uint32_t cluster_start, uint32_t *dir_n) {
  // TODO: figure out the length of the cluster chain (see
  // `get_cluster_chain_length`)
  uint32_t chain_length = get_cluster_chain_length(fs, cluster_start);

  // TODO: allocate a buffer large enough to hold the whole directory
  *dir_n = fs->sectors_per_cluster * chain_length * boot_sector.bytes_per_sec / sizeof(fat32_dirent_t);
  uint8_t *buf = kmalloc(fs->sectors_per_cluster * chain_length * boot_sector.bytes_per_sec);

  // TODO: read in the whole directory (see `read_cluster_chain`)
  read_cluster_chain(fs, cluster_start, buf);

  return (fat32_dirent_t *)buf;
}

pi_directory_t fat32_readdir(fat32_fs_t *fs, pi_dirent_t *dirent) {
  demand(init_p, "fat32 not initialized!");
  demand(dirent->is_dir_p, "tried to readdir a file!");
  // TODO: use `get_dirents` to read the raw dirent structures from the disk
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, dirent->cluster_id, &n_dirents);

  // TODO: allocate space to store the pi_dirent_t return values
  pi_dirent_t *pi_dirents = kmalloc(sizeof(pi_dirent_t) * n_dirents);

  // TODO: iterate over the directory and create pi_dirent_ts for every valid
  // file.  Don't include empty dirents, LFNs, or Volume IDs.  You can use
  // `dirent_convert`.
  uint32_t num_dirs = 0;
  for (uint32_t i = 0; i < n_dirents; i++) {
    // Check for an end of directory marker
    if (strlen(dirents[i].filename) == 0)
        break;
    
    // Skip free space, LFN version of name, and volume label
    if (fat32_dirent_free(&dirents[i]) || fat32_dirent_is_lfn(&dirents[i]) || dirents[i].attr & FAT32_VOLUME_LABEL) 
        continue; 

    // Valid directory found
    pi_dirents[num_dirs] = dirent_convert(dirents + i);
    num_dirs++;
  }

  // TODO: create a pi_directory_t using the dirents and the number of valid
  // dirents we found
  return (pi_directory_t) {
    .dirents = pi_dirents,
    .ndirents = num_dirs,
  };
}

static int find_dirent_with_name(fat32_dirent_t *dirents, int n, char *filename) {
  // TODO: iterate through the dirents, looking for a file which matches the
  // name; use `fat32_dirent_name` to convert the internal name format to a
  // normal string.
  for (int i = 0; i < n; i++) {
    char *dir_name = "";
    fat32_dirent_name(dirents + i, dir_name);
    if (strcmp(dir_name, filename) == 0)
        return i;
  }

  return -1;
}

pi_dirent_t *fat32_stat(fat32_fs_t *fs, pi_dirent_t *directory, char *filename) {
  demand(init_p, "fat32 not initialized!");
  demand(directory->is_dir_p, "tried to use a file as a directory");

  // TODO: use `get_dirents` to read the raw dirent structures from the disk
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  // TODO: Iterate through the directory's entries and find a dirent with the
  // provided name.  Return NULL if no such dirent exists.  You can use
  // `find_dirent_with_name` if you've implemented it.
  int dir_ent_idx = find_dirent_with_name(dirents, n_dirents, filename);
  if (dir_ent_idx == -1)
      return NULL;

  // TODO: allocate enough space for the dirent, then convert
  // (`dirent_convert`) the fat32 dirent into a Pi dirent.
  pi_dirent_t *dirent = kmalloc(sizeof(pi_dirent_t));
  *dirent = dirent_convert(&dirents[dir_ent_idx]);
  return dirent;
}

pi_file_t *fat32_read(fat32_fs_t *fs, pi_dirent_t *directory, char *filename) {
  // This should be pretty similar to readdir, but simpler.
  demand(init_p, "fat32 not initialized!");
  demand(directory->is_dir_p, "tried to use a file as a directory!");

  // TODO: read the dirents of the provided directory and look for one matching the provided name
  pi_dirent_t *f = fat32_stat(fs, directory, filename);

  // TODO: figure out the length of the cluster chain
  uint32_t chain_length = 0;
  if (f->nbytes > 0)
      chain_length = get_cluster_chain_length(fs, f->cluster_id);

  // TODO: allocate a buffer large enough to hold the whole file
  size_t n_alloc = fs->sectors_per_cluster * chain_length * boot_sector.bytes_per_sec;

  // TODO: read in the whole file (if it's not empty)
  char *data = "";
  if (n_alloc > 0) {
      data = kmalloc(n_alloc);
      read_cluster_chain(fs, f->cluster_id, data);
      // Ensure the file length is respected
      if (n_alloc > f->nbytes)
          data[f->nbytes] = '\0';
  }

  // TODO: fill the pi_file_t
  pi_file_t *file = kmalloc(sizeof(pi_file_t));
  *file = (pi_file_t) {
    .data = data,
    .n_data = f->nbytes,
    .n_alloc = n_alloc,
  };
  return file;
}

/******************************************************************************
 * Everything below here is for writing to the SD card (Part 7/Extension).  If
 * you're working on read-only code, you don't need any of this.
 ******************************************************************************/

static uint32_t find_free_cluster(fat32_fs_t *fs, uint32_t start_cluster) {
  // TODO: loop through the entries in the FAT until you find a free one
  // (fat32_fat_entry_type == FREE_CLUSTER).  Start from cluster 3.  Panic if
  // there are none left.
  if (start_cluster < 3) start_cluster = 3;

  uint32_t cluster = start_cluster;
  uint32_t n_entries = fs->n_entries;
  if (start_cluster < n_entries)
      n_entries -= start_cluster;

  while (cluster < n_entries) {
      if (fat32_fat_entry_type(fs->fat[cluster]) == FREE_CLUSTER)
          return cluster;

      cluster++;
  }

  if (trace_p) trace("failed to find free cluster from %d\n", start_cluster);
  panic("No more clusters on the disk!\n");
}

static void write_fat_to_disk(fat32_fs_t *fs) {
  // TODO: Write the FAT to disk.  In theory we should update every copy of the
  // FAT, but the first one is probably good enough.  A good OS would warn you
  // if the FATs are out of sync, but most OSes just read the first one without
  // complaining.
  if (trace_p) trace("syncing FAT\n");

  pi_sd_write(fs->fat, fs->fat_begin_lba, boot_sector.nsec_per_fat);
}

// Given the starting cluster index, write the data in `data` over the
// pre-existing chain, adding new clusters to the end if necessary.
static void write_cluster_chain(fat32_fs_t *fs, uint32_t start_cluster, uint8_t *data, uint32_t nbytes)
{
  // Walk the cluster chain in the FAT, writing the in-memory data to the
  // referenced clusters.  If the data is longer than the cluster chain, find
  // new free clusters and add them to the end of the list.
  // Things to consider:
  //  - what if the data is shorter than the cluster chain?
  //  - what if the data is longer than the cluster chain?
  //  - the last cluster needs to be marked LAST_CLUSTER
  //  - when do we want to write the updated FAT to the disk to prevent
  //  corruption?
  //  - what do we do when nbytes is 0?
  //  - what about when we don't have a valid cluster to start with?
  //
  //  This is the main "write" function we'll be using; the other functions
  //  will delegate their writing operations to this.

  // TODO: As long as we have bytes left to write and clusters to write them
  // to, walk the cluster chain writing them out.

  if (fat32_fat_entry_type(start_cluster) != USED_CLUSTER)
    // We should not write to this chain as it isn't valid
    return;

  uint32_t cluster = start_cluster;
  uint32_t bytes_per_cluster = fs->sectors_per_cluster * boot_sector.bytes_per_sec;
  /* Used at the end to decide if the last cluster should be free or not */
  uint32_t total_bytes = nbytes;

  // TODO: If we run out of clusters to write to, find free clusters using the
  // FAT and continue writing the bytes out.  Update the FAT to reflect the new
  // cluster.
  while (nbytes > 0)
  {
    pi_sd_write(data, cluster_to_lba(fs, cluster), fs->sectors_per_cluster);
    uint32_t bytes_written = bytes_per_cluster;
    if (bytes_written > nbytes)
      bytes_written = nbytes;

    nbytes -= bytes_written;
    data += bytes_written;

    // if there’s still data left, grow the chain one cluster at a time:
    if (nbytes > 0)
    {
      // if we were at the end of the chain, grab a new cluster
      uint32_t entry = fs->fat[cluster];
      if (fat32_fat_entry_type(entry) == LAST_CLUSTER)
      {
        uint32_t nxt = find_free_cluster(fs, 3) & CLUSTER_MASK;
        fs->fat[cluster] = nxt;      // link old→new
        fs->fat[nxt] = LAST_CLUSTER; // new end-of-chain
      }
      // move into the newly linked cluster
      cluster = fs->fat[cluster];
    }
  }

  // TODO: If we run out of bytes to write before using all the clusters, mark
  // the final cluster as "LAST_CLUSTER" in the FAT, then free all the clusters
  // later in the chain.
  uint32_t last_cluster = cluster;
  while (fat32_fat_entry_type(cluster) == USED_CLUSTER)
  {
    uint32_t next_cluster = fs->fat[cluster];
    fs->fat[cluster] = FREE_CLUSTER;
    cluster = next_cluster;
  }

  // TODO: Ensure that the last cluster in the chain is marked "LAST_CLUSTER".
  // The one exception to this is if we're writing 0 bytes in total, in which
  // case we don't want to use any clusters at all.
  if (total_bytes > 0)
    fs->fat[last_cluster] = LAST_CLUSTER;
  else
    fs->fat[last_cluster] = FREE_CLUSTER;

  write_fat_to_disk(fs);
}

int fat32_rename(fat32_fs_t *fs, pi_dirent_t *directory, char *oldname, char *newname) {
  // TODO: Get the dirents `directory` off the disk, and iterate through them
  // looking for the file.  When you find it, rename it and write it back to
  // the disk (validate the name first).  Return 0 in case of an error, or 1
  // on success.
  // Consider:
  //  - what do you do when there's already a file with the new name?
  demand(init_p, "fat32 not initialized!");
  if (trace_p) trace("renaming %s to %s\n", oldname, newname);
  if (!fat32_is_valid_name(newname)) return 0;

  // TODO: get the dirents and find the right one
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  int dir_ent_idx_old = find_dirent_with_name(dirents, n_dirents, oldname);
  if (dir_ent_idx_old == -1) {
      if (trace_p) trace("no file found with name %s\n", oldname);
      return 0;
  }

  int dir_ent_idx_new = find_dirent_with_name(dirents, n_dirents, newname);
  if (dir_ent_idx_new != -1) {
    if (trace_p) trace("file already exists with name %s. Overwriting\n", newname);
    if (!fat32_delete(fs, directory, newname))
        return 0;

    // The file was deleted, so re-retrieve the directory entries
    // TODO: Better way or error handling?
    dirents = get_dirents(fs, directory->cluster_id, &n_dirents);
    dir_ent_idx_old = find_dirent_with_name(dirents, n_dirents, oldname);
  }

  // TODO: update the dirent's name
  fat32_dirent_set_name(&dirents[dir_ent_idx_old], newname);

  // TODO: write out the directory, using the existing cluster chain (or
  // appending to the end); implementing `write_cluster_chain` will help
  write_cluster_chain(fs, directory->cluster_id, (uint8_t *)dirents, n_dirents * sizeof(fat32_dirent_t));
  return 1;
}

// Create a new directory entry for an empty file (or directory).
pi_dirent_t *fat32_create(fat32_fs_t *fs, pi_dirent_t *directory, char *filename, int is_dir) {
  demand(init_p, "fat32 not initialized!");
  if (trace_p) trace("creating %s\n", filename);
  if (!fat32_is_valid_name(filename)) return NULL;

  // TODO: read the dirents and make sure there isn't already a file with the
  // same name
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  int dir_ent_idx = find_dirent_with_name(dirents, n_dirents, filename);
  if (dir_ent_idx != -1)
      return NULL;

  // TODO: look for a free directory entry and use it to store the data for the
  // new file.  If there aren't any free directory entries, either panic or
  // (better) handle it appropriately by extending the directory to a new
  // cluster.
  // When you find one, update it to match the name and attributes
  // specified; set the size and cluster to 0.
  fat32_dirent_t *free_dirent = NULL;
  for (uint32_t i = 0; i < n_dirents; i++) {
    if (fat32_dirent_free(dirents + i)) {
      free_dirent = dirents + i;
      break;
    }
  }

  if (!free_dirent)
      // TODO (Joey): extend directory instead of panicing
      panic("out of dir entries!\n");

  fat32_dirent_set_name(free_dirent, filename);
  free_dirent->file_nbytes = 0;
  free_dirent->hi_start = 0;
  free_dirent->lo_start = 0;
  if (is_dir)
      free_dirent->attr = FAT32_DIR;

  // TODO: write out the updated directory to the disk
  write_cluster_chain(fs, directory->cluster_id, (uint8_t *)dirents, n_dirents * sizeof(fat32_dirent_t));

  // TODO: convert the dirent to a `pi_dirent_t` and return a (kmalloc'ed)
  // pointer
  pi_dirent_t *dirent = kmalloc(sizeof(pi_dirent_t));
  *dirent = dirent_convert(free_dirent);
  return dirent;
}

// Delete a file, including its directory entry.
int fat32_delete(fat32_fs_t *fs, pi_dirent_t *directory, char *filename) {
  demand(init_p, "fat32 not initialized!");
  if (trace_p) trace("deleting %s\n", filename);
  if (!fat32_is_valid_name(filename)) return 0;
  // TODO: look for a matching directory entry, and set the first byte of the
  // name to 0xE5 to mark it as free
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  int dir_ent_idx = find_dirent_with_name(dirents, n_dirents, filename);
  if (dir_ent_idx == -1)
      return 0;

  fat32_dirent_t *dirent = &dirents[dir_ent_idx];
  dirent->filename[0] = 0xe5;

  // TODO: free the clusters referenced by this dirent
  uint32_t cluster = fat32_cluster_id(dirent);
  while (fat32_fat_entry_type(fs->fat[cluster]) == USED_CLUSTER) {
      uint32_t next_cluster = fs->fat[cluster];
      fs->fat[cluster] = FREE_CLUSTER;
      cluster = next_cluster;
  }

  // Last cluster of the file. Only free if the current cluster isn't free already (will only happen
  // when deleting empty files)
  if (fat32_fat_entry_type(cluster) != FREE_CLUSTER)
      fs->fat[cluster] = FREE_CLUSTER;

  write_fat_to_disk(fs);

  // TODO: write out the updated directory to the disk
  write_cluster_chain(fs, directory->cluster_id, (uint8_t *)dirents, n_dirents * sizeof(fat32_dirent_t));
  return 1;
}

int fat32_truncate(fat32_fs_t *fs, pi_dirent_t *directory, char *filename, unsigned length) {
  demand(init_p, "fat32 not initialized!");
  if (trace_p) trace("truncating %s\n", filename);

  // TODO: edit the directory entry of the file to list its length as `length` bytes,
  // then modify the cluster chain to either free unused clusters or add new
  // clusters.
  //
  // Consider: what if the file we're truncating has length 0? what if we're
  // truncating to length 0?
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  int dir_ent_idx = find_dirent_with_name(dirents, n_dirents, filename);
  if (dir_ent_idx == -1)
      return 0;

  fat32_dirent_t *dirent = &dirents[dir_ent_idx];
  uint32_t old_nbytes = dirent->file_nbytes;

  // Optimization, but not necessary for correctness
  if (old_nbytes == length) {
      trace("Truncating file to same length. Not touching\n");
      return 1;
  }

  dirent->file_nbytes = length;
  trace("Truncating file from %d to %d\n", old_nbytes, length);

  if (old_nbytes <= length) {
    // Truncating to a larger (or equal) file, pad with zero bytes
    pi_file_t *file = fat32_read(fs, directory, filename);
    char *new_data = kmalloc(length);
    strcpy(new_data, file->data);
    memset(new_data + old_nbytes, '\0', length - old_nbytes);

    pi_file_t new_f = (pi_file_t) {
        .data = new_data,
        .n_data = length,
        .n_alloc = length,
    };
    return fat32_write(fs, directory, filename, &new_f);
  }

  // Truncate to a smaller file, free clusters if necessary.
  // Unoptimized way: Read all data and only write the truncated value (would handle edge case but
  // is inefficient because we don't need to read and write, just adjust the FAT)

  // Optimized way: Just modify the dirent and fat (truncated data is lost but still potentiall
  // recoverable)
  
  uint32_t bytes_per_cluster = fs->sectors_per_cluster * boot_sector.bytes_per_sec;
  uint32_t cluster = fat32_cluster_id(dirent);

  // First, find the last used cluster in the new file
  uint32_t cur_bytes = bytes_per_cluster;
  while (cur_bytes < length) {
      cluster = fs->fat[cluster];
      cur_bytes += bytes_per_cluster;
  }
  
  // Below is the same logic used as at the end of `write_cluster_chain`

  // Now that we have found the last cluster of the file, free the clusters after it
  uint32_t last_cluster = cluster;
  while (fat32_fat_entry_type(cluster) == USED_CLUSTER) {
    uint32_t next_cluster = fs->fat[cluster];
    fs->fat[cluster] = FREE_CLUSTER;
    cluster = next_cluster;
  }

  // Set the last cluster depending on the length of the new file
  if (length > 0)
      fs->fat[last_cluster] = LAST_CLUSTER;
  else {
      fs->fat[last_cluster] = FREE_CLUSTER;
      // Need to remove cluster as this is now an empty file
      dirent->hi_start = 0;
      dirent->lo_start = 0;
  }

  // Write out the directory entry
  write_cluster_chain(fs, directory->cluster_id, (uint8_t *)dirents, n_dirents * sizeof(fat32_dirent_t));
  return 1;
}

int fat32_write(fat32_fs_t *fs, pi_dirent_t *directory, char *filename, pi_file_t *file) {
  demand(init_p, "fat32 not initialized!");
  demand(directory->is_dir_p, "tried to use a file as a directory!");

  // TODO: Surprisingly, this one should be rather straightforward now.
  // - load the directory
  // - exit with an error (0) if there's no matching directory entry
  // - update the directory entry with the new size
  // - write out the file as clusters & update the FAT
  // - write out the directory entry
  // Special case: the file is empty to start with, so we need to update the
  // start cluster in the dirent

  // Get the directory entries
  uint32_t n_dirents;
  fat32_dirent_t *dirents = get_dirents(fs, directory->cluster_id, &n_dirents);

  // Find the matching directory entry. Error if it isn't found
  int dir_ent_idx = find_dirent_with_name(dirents, n_dirents, filename);
  if (dir_ent_idx == -1)
      return 0;

  fat32_dirent_t *dirent = &dirents[dir_ent_idx];

  uint32_t cluster = fat32_cluster_id(dirent);
  if (file->n_data == 0) {
      // No data to write, so set the directory entry to be empty
      trace("File to write has no data. Clearing file\n");
      dirent->hi_start = 0;
      dirent->lo_start = 0;
  } else if (fat32_fat_entry_type(cluster) == FREE_CLUSTER) {
    // File is empty, so update the dirent with a new cluster
    cluster = find_free_cluster(fs, 3) & CLUSTER_MASK;
    // Complete the, so far empty, chain
    fs->fat[cluster] = LAST_CLUSTER;

    trace("empty file, updating directory cluster number\n");
    dirent->hi_start = cluster >> 16;
    dirent->lo_start = cluster & 0xFFFF; 
  }

  // Write out the file as clusters & update the FAT
  trace("writing file\n");
  write_cluster_chain(fs, cluster, file->data, file->n_data);

  // Update the directory entry with the new size
  dirent->file_nbytes = file->n_data;

  // Write out the directory entry
  trace("writing dirent\n");
  write_cluster_chain(fs, directory->cluster_id, (uint8_t *)dirents, n_dirents * sizeof(fat32_dirent_t));
  return 1;
}

int fat32_flush(fat32_fs_t *fs) {
  demand(init_p, "fat32 not initialized!");
  // no-op
  return 0;
}
